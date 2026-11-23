"""Score middle windows with the existing XLM-R checkpoint."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from lab import LABELS, load, partitions, save_metrics, scores, write_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "split", "checkpoint", "reference", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--length", type=int, default=384)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    if args.batch < 1 or args.length < 8:
        parser.error("Use a positive batch and a length of at least eight")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run")
    if args.out.exists():
        raise FileExistsError(args.out)

    text, _, val, _ = partitions(args)
    test = load(args.data / "test.csv", ["id", text])
    references = []
    for name, frame in (("val", val), ("test", test)):
        ref = load(args.reference / f"{name}.csv", ["id", *LABELS])
        if set(ref.index) != set(frame.index):
            raise ValueError(f"Reference {name} IDs do not match")
        values = ref.loc[frame.index, LABELS].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(1), 1):
            raise ValueError(f"Invalid reference {name} scores")
        references.append(values)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, local_files_only=True, low_cpu_mem_usage=True,
    ).to("cuda").eval()
    if [model.config.id2label[i] for i in range(len(LABELS))] != LABELS:
        raise ValueError("Checkpoint class order differs")
    if args.length > tokenizer.model_max_length:
        raise ValueError("Window exceeds the tokenizer's maximum length")
    budget = args.length - tokenizer.num_special_tokens_to_add(pair=False)
    collate = DataCollatorWithPadding(tokenizer)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def predict(frame, reference, name):
        result = reference.copy()
        long_docs = np.zeros(len(frame), dtype=bool)
        for start in range(0, len(frame), 256):
            docs = frame.iloc[start:start + 256][text].astype(str).tolist()
            encoded = tokenizer(docs, add_special_tokens=False, truncation=False, verbose=False)
            rows, features = [], []
            for offset, ids in enumerate(encoded.input_ids):
                if len(ids) <= budget:
                    continue
                center = (len(ids) - budget) // 2
                window = tokenizer.build_inputs_with_special_tokens(ids[center:center + budget])
                rows.append(start + offset)
                features.append({"input_ids": window, "attention_mask": [1] * len(window)})
            for offset in range(0, len(rows), args.batch):
                row_slice = rows[offset:offset + args.batch]
                batch = collate(features[offset:offset + args.batch]).to("cuda")
                result[row_slice] = softmax(model(**batch).logits.float().cpu().numpy(), axis=1)
                long_docs[row_slice] = True
            if start % 8192 == 0:
                print(f"{name}: {min(start + len(docs), len(frame))}/{len(frame)}", flush=True)
        return result, long_docs

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        val_prob, val_long = predict(val, references[0], "val")
        test_prob, test_long = predict(test, references[1], "test")

    args.out.mkdir(parents=True, exist_ok=False)
    middle = write_predictions(args.out, val, val_prob, test, test_prob)
    variants = {"head_tail": scores(val.label, references[0]), "middle": middle}
    for alpha in (0.1, 0.2, 0.3):
        name = f"middle_weight_{alpha}"
        target = args.out / name
        target.mkdir()
        variants[name] = write_predictions(
            target, val, (1 - alpha) * references[0] + alpha * val_prob,
            test, (1 - alpha) * references[1] + alpha * test_prob,
        )
    diagnostics = {}
    for name, mask in (("short", ~val_long), ("long", val_long)):
        if mask.any():
            diagnostics[name] = {
                "rows": int(mask.sum()),
                "head_tail": scores(val.label.iloc[mask], references[0][mask]),
                "middle": scores(val.label.iloc[mask], val_prob[mask]),
                "middle_weight_0.2": scores(
                    val.label.iloc[mask], 0.8 * references[0][mask] + 0.2 * val_prob[mask]
                ),
            }
    pd.DataFrame({"id": val.index, "long": val_long}).to_csv(args.out / "val_lengths.csv", index=False)
    save_metrics(args, variants, diagnostics=diagnostics, test_long_rows=int(test_long.sum()))


if __name__ == "__main__":
    main()
