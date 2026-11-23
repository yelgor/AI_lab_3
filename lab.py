import argparse
import hashlib
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy.sparse import csr_matrix, hstack
from scipy.special import softmax

LABELS = ["ai", "fusion", "human"]
SEED = 42


def load(path, required):
    frame = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
    if not set(required) <= set(frame):
        raise ValueError(f"{path} must contain {required}; found {list(frame)}")
    if frame.id.eq("").any() or not frame.id.is_unique:
        raise ValueError(f"{path} contains missing or repeated IDs")
    return frame.set_index("id", drop=False)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized(text):
    return " ".join(str(text).casefold().split())


def split_groups(train, text, group):
    # Connect repeated text and known source groups, including transitive links.
    parents = list(range(len(train)))
    seen = {}

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    sources = train[group] if group else [""] * len(train)
    for index, (document, source) in enumerate(zip(train[text], sources)):
        for key in (("text", normalized(document)), ("source", str(source).strip())):
            if key[1]:
                parents[root(index)] = root(seen.setdefault(key, index))
    return [root(index) for index in range(len(train))]


def prepare(args):
    path = args.data / "train.csv"
    required = ["id", args.text, "label"] + ([args.group] if args.group else [])
    train = load(path, required)
    if set(train.label) != set(LABELS):
        raise ValueError(f"Expected labels {LABELS}")
    groups = split_groups(train, args.text, args.group)
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    fit, val = next(splitter.split(train, train.label, groups))
    if not all(set(train.iloc[rows].label) == set(LABELS) for rows in (fit, val)):
        raise ValueError("A split partition is missing a class")
    if args.sample < len(fit):
        sample, _ = train_test_split(
            fit, train_size=args.sample, stratify=train.iloc[fit].label, random_state=SEED
        )
    else:
        sample = fit
    manifest = {
        "sha256": digest(path), "text": args.text, "group": args.group,
        "fit": train.index[fit].tolist(), "val": train.index[val].tolist(),
        "encoder": train.index[sample].tolist(),
    }
    args.split.parent.mkdir(parents=True, exist_ok=True)
    with args.split.open("x") as stream:
        json.dump(manifest, stream)
    print(f"Fit {len(fit)}, validation {len(val)}, encoder fit {len(sample)}")


def partitions(args):
    manifest = json.loads(args.split.read_text())
    path = args.data / "train.csv"
    if digest(path) != manifest["sha256"]:
        raise ValueError("Training CSV differs from the frozen split")
    text = manifest["text"]
    train = load(path, ["id", text, "label"])
    return text, *(train.loc[manifest[key]] for key in ("fit", "val", "encoder"))


def eda(args):
    text, fit, _, _ = partitions(args)
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    lengths = fit[text].astype(str).str.len()
    lengths.groupby(fit.label).quantile([0, .25, .5, .75, .95, 1]).unstack().to_csv(out / "lengths.csv")
    fit.label.value_counts().to_csv(out / "classes.csv")
    for name, column in (("language", args.language), ("domain", args.domain)):
        if column:
            pd.crosstab(fit[column], fit.label).to_csv(out / f"{name}.csv")
    examples = fit.groupby("label").sample(n=2, random_state=SEED)[["id", "label", text]].copy()
    examples[text] = examples[text].astype(str).str.slice(0, 400)
    examples.to_csv(out / "examples.csv", index=False)


def style(text):
    words = str(text).split()
    thirds = [words[len(words) * i // 3:len(words) * (i + 1) // 3] for i in range(3)]

    def stats(tokens):
        segment = " ".join(tokens)
        sentences = [part.split() for part in re.split(r"[.!?]+", segment) if part.strip()]
        lengths = [len(sentence) for sentence in sentences] or [0]
        rates = [100 * sum(segment.count(char) for char in chars) / max(1, len(tokens))
                 for chars in [",;:", "-–—", "()[]", "!?"]]
        return [np.mean(lengths), np.std(lengths), *rates]

    local = np.array([stats(part) for part in thirds if len(part) >= 10])
    variation = np.ptp(local, axis=0) if len(local) >= 2 else np.zeros(6)
    return [np.log1p(len(words)), *stats(words), *variation]


def scores(labels, probabilities):
    predicted = np.array(LABELS)[probabilities.argmax(axis=1)]
    return {
        "macro_f1": f1_score(labels, predicted, labels=LABELS, average="macro"),
        "per_class": classification_report(
            labels, predicted, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=LABELS).tolist(),
        "class_order": LABELS,
    }


def write_predictions(out, val, val_prob, test, test_prob):
    for name, frame, probabilities in (("val", val, val_prob), ("test", test, test_prob)):
        result = pd.DataFrame(probabilities, columns=LABELS)
        result.insert(0, "id", frame.index)
        result.to_csv(out / f"{name}.csv", index=False)
    return scores(val.label, val_prob)


def baseline(args):
    text, fit, val, _ = partitions(args)
    test = load(args.data / "test.csv", ["id", text])
    args.out.mkdir(parents=True, exist_ok=False)
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=args.features,
                           sublinear_tf=True, dtype=np.float32)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3,
                           max_features=args.features, sublinear_tf=True, dtype=np.float32,
                           lowercase=False)
    # Cap all documents uniformly to keep CPU memory bounded.
    documents = [frame[text].astype(str).str.slice(0, args.max_chars) for frame in (fit, val, test)]
    word.fit(documents[0])
    char.fit(documents[0])
    matrices = [hstack([word.transform(rows), char.transform(rows)], format="csr")
                for rows in documents]
    best = -1
    reference_c = None
    results = {}
    for c in args.c:
        name = f"tfidf-c{c:g}"
        model = LinearSVC(C=c, dual="auto", random_state=SEED)
        model.fit(matrices[0], fit.label)
        prob = softmax(model.decision_function(matrices[1]), axis=1)
        results[name] = scores(val.label, prob)
        if results[name]["macro_f1"] > best:
            best = results[name]["macro_f1"]
            reference_c = c
            reference_prob = prob
            reference_model = model
    reference = args.out / "tfidf"
    reference.mkdir()
    write_predictions(reference, val, reference_prob, test,
                      softmax(reference_model.decision_function(matrices[2]), axis=1))
    names = np.concatenate([word.get_feature_names_out(), char.get_feature_names_out()])
    attribution = {}
    for label, weights in zip(reference_model.classes_, reference_model.coef_):
        top = np.argsort(weights)[-15:][::-1]
        attribution[label] = [
            {"type": "word" if index < len(word.vocabulary_) else "char",
             "ngram": str(names[index]), "weight": float(weights[index])} for index in top
        ]
    (reference / "top_coefficients.json").write_text(json.dumps(attribution, indent=2))
    style_matrices = [np.array([style(row) for row in frame[text]], dtype=np.float32)
                      for frame in (fit, val, test)]
    scaler = StandardScaler().fit(style_matrices[0])
    extended = [
        hstack([matrix, csr_matrix(scaler.transform(dense) * args.style_scale)], format="csr")
        for matrix, dense in zip(matrices, style_matrices)
    ]
    model = LinearSVC(C=reference_c, dual="auto", random_state=SEED)
    model.fit(extended[0], fit.label)
    target = args.out / "stylometry"
    target.mkdir()
    results["stylometry"] = write_predictions(
        target, val, softmax(model.decision_function(extended[1]), axis=1),
        test, softmax(model.decision_function(extended[2]), axis=1),
    )
    save_metrics(args, results, reference_c=reference_c)


def save_metrics(args, results, **extra):
    config = {name: value for name, value in vars(args).items() if name != "func"}
    (args.out / "metrics.json").write_text(json.dumps(
        {"config": config, "split_sha256": digest(args.split), "results": results, **extra},
        indent=2, default=str
    ))
    for name, metrics in results.items():
        print(f"{name}: {metrics['macro_f1']:.5f}")


def select(args):
    _, _, val, _ = partitions(args)
    sample = load(args.data / "sample_submission.csv", ["id", "label"])
    test = load(args.data / "test.csv", ["id"])
    if len(test) != args.expected or set(sample.index) != set(test.index):
        raise ValueError("Test size or sample ID set does not match the expected submission")
    args.out.mkdir(parents=True, exist_ok=False)
    pairs = []
    for path in args.runs:
        blocks = []
        for name, index in (("val", val.index), ("test", sample.index)):
            frame = load(path / f"{name}.csv", ["id", *LABELS])
            if set(frame.index) != set(index):
                raise ValueError(f"Prediction IDs differ in {path / (name + '.csv')}")
            values = frame.loc[index, LABELS].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(1), 1):
                raise ValueError(f"Invalid probabilities in {path}")
            blocks.append(values)
        pairs.append(blocks)

    candidates = []

    def add(weights, fusion_bias=0.0):
        val_prob = sum(w * pair[0] for w, pair in zip(weights, pairs))
        val_prob[:, LABELS.index("fusion")] *= np.exp(fusion_bias)
        predicted = np.array(LABELS)[val_prob.argmax(axis=1)]
        metric = f1_score(val.label, predicted, labels=LABELS, average="macro")
        candidates.append((metric, tuple(weights), fusion_bias))

    weights = np.eye(len(pairs))
    for weight in weights:
        add(weight)
    add(weights.mean(axis=0))
    if len(pairs) > 1:
        ranked = sorted(range(len(pairs)), key=lambda i: candidates[i][0], reverse=True)
        model_pairs = combinations(ranked, 2) if args.all_pairs else [tuple(ranked[:2])]
        for first, second in model_pairs:
            for alpha in (.25, .5, .75):
                add(alpha * weights[first] + (1 - alpha) * weights[second])
    best = max(candidates, key=lambda candidate: candidate[0])
    for bias in (-.25, -.1, .1, .25):
        add(best[1], bias)
    ordered = sorted(candidates, key=lambda candidate: candidate[0], reverse=True)
    _, weight, bias = ordered[0]
    val_prob = sum(w * pair[0] for w, pair in zip(weight, pairs))
    val_prob[:, LABELS.index("fusion")] *= np.exp(bias)
    result = scores(val.label, val_prob)
    probs = sum(w * pair[1] for w, pair in zip(weight, pairs))
    probs[:, LABELS.index("fusion")] *= np.exp(bias)
    predicted = np.array(LABELS)[probs.argmax(axis=1)]
    pd.DataFrame({"id": sample.index, "label": predicted}).to_csv(
        args.out / "submission_1.csv", index=False
    )
    results = {"submission_1": {
        **result, "weights": dict(zip(map(str, args.runs), weight)), "fusion_bias": bias,
    }}
    save_metrics(args, results, candidates=[
        {"macro_f1": metric, "weights": list(weight), "fusion_bias": bias}
        for metric, weight, bias in ordered
    ])


def encoder(args):
    from encoder import run

    run(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--split", type=Path, default=Path("runs/split.json"))
    commands = parser.add_subparsers(required=True)

    def command(name, func, default_out=None):
        sub = commands.add_parser(name)
        sub.set_defaults(func=func)
        if default_out:
            sub.add_argument("--out", type=Path, default=Path(default_out))
        return sub

    prep = command("prepare", prepare)
    prep.add_argument("--text", default="text")
    prep.add_argument("--group")
    prep.add_argument("--sample", type=int, default=30000)
    analysis = command("eda", eda, "runs/eda")
    analysis.add_argument("--language")
    analysis.add_argument("--domain")
    linear = command("baseline", baseline, "runs/baseline")
    linear.add_argument("--features", type=int, default=100000)
    linear.add_argument("--max-chars", type=int, default=12000)
    linear.add_argument("--c", type=float, nargs="+", default=[.25, 1, 4])
    linear.add_argument("--style-scale", type=float, default=.1)
    transformer = command("encoder", encoder, "runs/encoder")
    transformer.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    transformer.add_argument("--revision", default="main")
    transformer.add_argument("--strategy", choices=["head", "head-tail"], default="head")
    transformer.add_argument("--length", type=int, default=256)
    transformer.add_argument("--batch", type=int, default=8)
    transformer.add_argument("--accumulation", type=int, default=4)
    transformer.add_argument("--epochs", type=int, default=2)
    transformer.add_argument("--lr", type=float, default=2e-5)
    transformer.add_argument("--weighted", action="store_true")
    transformer.add_argument("--fit-rows", type=int)
    transformer.add_argument("--gradient-checkpointing", action="store_true")
    ensemble = command("select", select, "runs/final")
    ensemble.add_argument("runs", nargs="+", type=Path)
    ensemble.add_argument("--expected", type=int, default=25000)
    ensemble.add_argument("--all-pairs", action="store_true")
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
