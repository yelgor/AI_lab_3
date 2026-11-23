import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from encoder import EncodedDocuments
from lab import LABELS, load, partitions, save_metrics, write_predictions

parser = argparse.ArgumentParser()
for name in ("data", "split", "checkpoint", "out"):
    parser.add_argument(f"--{name}", type=Path, required=True)
parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
args = parser.parse_args()
text, _, val, _ = partitions(args)
test = load(args.data / "test.csv", ["id", text])
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    args.checkpoint, local_files_only=True, low_cpu_mem_usage=True,
).to(args.device).eval()
assert [model.config.id2label[i] for i in range(len(LABELS))] == LABELS
collate = DataCollatorWithPadding(tokenizer)

def predict(frame, name):
    outputs = []
    for start in range(0, len(frame), 256):
        docs = EncodedDocuments(frame.iloc[start:start + 256], text, tokenizer, 384, "head-tail", False)
        for offset in range(0, len(docs), 8):
            batch = collate(docs.records[offset:offset + 8]).to(args.device)
            outputs.append(model(**batch).logits.float().cpu().numpy())
        if start % 1024 == 0:
            print(f"{name}: {min(start + 256, len(frame))}/{len(frame)}")
    return softmax(np.concatenate(outputs), axis=1)

args.out.mkdir(parents=True)
precision = torch.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else nullcontext()
with torch.inference_mode(), precision:
    result = write_predictions(args.out, val, predict(val, "val"), test, predict(test, "test"))
save_metrics(args, {"epoch1": result})
