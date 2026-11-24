"""Export aggregate EDA and validation metrics from local competition artifacts."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from lab import LABELS, SEED, digest, partitions


def table(frame, index_name):
    def row(values):
        return "| " + " | ".join(str(value).replace("|", "&#124;").replace("\n", " ") for value in values) + " |"

    return "\n".join([
        row([index_name, *frame.columns]),
        row(["---"] * (len(frame.columns) + 1)),
        *(row([index, *values]) for index, values in zip(frame.index, frame.to_numpy())),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--result", default="submission_1")
    parser.add_argument("--language")
    parser.add_argument("--domain")
    parser.add_argument("--example-chars", type=int, default=240)
    parser.add_argument("--out", type=Path, default=Path("appendix"))
    args = parser.parse_args()

    if args.example_chars < 1:
        parser.error("--example-chars must be positive")

    metrics = json.loads(args.metrics.read_text())
    if metrics["split_sha256"] != digest(args.split):
        parser.error("Metrics were produced with a different split")
    record = metrics["results"][args.result]
    column, fit, val, _ = partitions(args)
    labels = record["class_order"]
    matrix = pd.DataFrame(record["confusion_matrix"], index=labels, columns=labels)
    if set(labels) != set(LABELS) or not matrix.sum(axis=1).equals(
        val.label.value_counts().reindex(labels, fill_value=0)
    ):
        parser.error("Validation class counts do not match the confusion matrix")
    report = pd.DataFrame.from_dict(
        {label: record["per_class"][label] for label in labels}, orient="index"
    )
    report["support"] = report["support"].astype(int)
    for metric in ("precision", "recall", "f1-score"):
        report[metric] = report[metric].map("{:.5f}".format)
    length = fit[column].astype(str).str.len()
    by_class = fit.label.value_counts().reindex(labels).to_frame("Кількість")
    summary = length.groupby(fit.label).quantile([.25, .5, .75, .95]).unstack().reindex(labels)
    summary.columns = ["P25", "Медіана", "P75", "P95"]
    summary = summary.round().astype(int)

    metadata = []
    for key, title, explicit, aliases in (
        ("language", "Мова", args.language, ("language", "lang")),
        ("domain", "Домен", args.domain, ("domain", "genre")),
    ):
        name = explicit or next((alias for alias in aliases if alias in fit), None)
        if name:
            if name not in fit:
                parser.error(f"Column {name!r} is absent. Available columns: {list(fit.columns)}")
            counts = pd.crosstab(fit[name].replace("", "(невідомо)"), fit.label)
            metadata.append((key, title, counts.reindex(columns=labels, fill_value=0)))

    args.out.mkdir(parents=True, exist_ok=False)
    sections = [
        "# Додаток", "## EDA навчальної частини",
        table(by_class.astype(str), "Клас"), "### Довжина текстів у символах",
        table(summary.astype(str), "Клас"), "![Квантилі довжин](lengths.svg)",
    ]
    ax = summary.plot.bar(rot=0, figsize=(7, 4))
    ax.set(xlabel="Клас", ylabel="Символи")
    ax.figure.tight_layout()
    ax.figure.savefig(args.out / "lengths.svg")
    plt.close(ax.figure)

    for key, title, counts in metadata:
        ax = counts.plot.barh(stacked=True, figsize=(8, max(3, .4 * len(counts))))
        ax.set(xlabel="Тексти", ylabel=title)
        ax.figure.tight_layout()
        ax.figure.savefig(args.out / f"{key}.svg")
        plt.close(ax.figure)
        sections.extend([f"### {title}", table(counts.astype(str), title), f"![{title}]({key}.svg)"])
    if len(metadata) < 2:
        missing = {"language": "мова", "domain": "домен"}
        for key, _, _ in metadata:
            missing.pop(key)
        sections.append("Метадані відсутні або колонки не вказано: " + ", ".join(missing.values()) + ".")

    examples = fit.loc[length.gt(0)].groupby("label").sample(n=1, random_state=SEED)
    snippets = examples.set_index("label")[column].astype(str).str.split().str.join(" ")
    snippets = snippets.reindex(labels)
    clipped = snippets.str.len().gt(args.example_chars)
    snippets = snippets.str.slice(0, args.example_chars).where(
        ~clipped, snippets.str.slice(0, args.example_chars - 1) + "…"
    )
    sections.extend([
        "### Приклади з навчальної частини",
        "Мітки взято з набору даних. Для fusion розмітка задається на рівні документа.",
        table(snippets.fillna("Порожній клас").to_frame("Уривок"), "Клас"),
        "## Метрики фінального варіанта на валідації",
        f"Macro-F1 — {record['macro_f1']:.5f}.",
        table(report.astype(str), "Клас"),
        "### Матриця помилок", "Рядки — справжні класи. Стовпці — прогнози.",
        table(matrix.astype(str), "Факт / прогноз"),
    ])
    (args.out / "APPENDIX.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(args.out / "APPENDIX.md")


if __name__ == "__main__":
    main()
