"""Train a bounded-memory word/character hashing baseline on the frozen split."""
import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from scipy.special import softmax
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from lab import LABELS, SEED, load, partitions, save_metrics, write_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--split", type=Path, default=Path("runs/split.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fit-rows", type=int, default=60000)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    if min(args.fit_rows, args.max_chars, args.batch, args.epochs) < 1:
        parser.error("Numeric parameters must be positive")

    column, fit, val, _ = partitions(args)
    if args.fit_rows < len(fit):
        fit, _ = train_test_split(
            fit, train_size=args.fit_rows, stratify=fit.label, random_state=SEED
        )
    test = load(args.data / "test.csv", ["id", column])
    args.out.mkdir(parents=True, exist_ok=False)
    print(f"Training on {len(fit)} documents", flush=True)

    buckets = 2 ** 19
    word = HashingVectorizer(
        n_features=buckets, ngram_range=(1, 2), norm=None,
        alternate_sign=False, dtype=np.float32,
    )
    char = HashingVectorizer(
        n_features=buckets, analyzer="char", ngram_range=(3, 5), norm=None,
        alternate_sign=False, lowercase=False, dtype=np.float32,
    )

    def batches(frame):
        for start in range(0, len(frame), args.batch):
            yield frame.iloc[start:start + args.batch]

    def counts(frame):
        docs = frame[column].astype(str).str.slice(0, args.max_chars)
        return [word.transform(docs), char.transform(docs)]

    # CSR has one stored entry per occupied bucket per document.
    df = [np.zeros(buckets, dtype=np.int64) for _ in range(2)]
    for batch in batches(fit):
        for freq, matrix in zip(df, counts(batch)):
            np.add.at(freq, matrix.indices, 1)
    idf = [np.asarray(np.log((1 + len(fit)) / (1 + freq)) + 1, dtype=np.float32)
           for freq in df]

    def features(frame):
        matrices = counts(frame)
        for matrix, weights in zip(matrices, idf):
            matrix.data[:] = (1 + np.log(matrix.data)) * weights[matrix.indices]
            normalize(matrix, copy=False)
        return hstack(matrices, format="csr")

    model = SGDClassifier(
        loss="modified_huber", alpha=1e-5, average=True, random_state=SEED
    )
    rng = np.random.default_rng(SEED)
    for epoch in range(args.epochs):
        for batch in batches(fit.iloc[rng.permutation(len(fit))]):
            model.partial_fit(features(batch), batch.label, classes=LABELS)
        print(f"Completed epoch {epoch + 1}/{args.epochs}", flush=True)
    if list(model.classes_) != LABELS:
        raise ValueError("Unexpected model class order")

    def predict(frame):
        return np.concatenate([
            softmax(model.decision_function(features(batch)), axis=1)
            for batch in batches(frame)
        ])

    metrics = write_predictions(args.out, val, predict(val), test, predict(test))
    save_metrics(args, {"hashed": metrics}, fit_rows=len(fit))


if __name__ == "__main__":
    main()
