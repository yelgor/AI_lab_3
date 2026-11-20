import numpy as np
import torch
from scipy.special import softmax
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    Trainer, TrainingArguments, set_seed,
)

from lab import LABELS, SEED, load, partitions, save_metrics, scores, write_predictions


class EncodedDocuments(Dataset):
    def __init__(self, frame, column, tokenizer, length, strategy, labeled):
        documents = frame[column].astype(str).tolist()
        self.records = []
        if strategy == "head":
            for start in range(0, len(documents), 256):
                items = tokenizer(documents[start:start + 256], truncation=True, max_length=length)
                self.records.extend(
                    {"input_ids": ids, "attention_mask": mask}
                    for ids, mask in zip(items.input_ids, items.attention_mask)
                )
        else:
            budget = length - tokenizer.num_special_tokens_to_add(pair=True)
            if budget < 2:
                raise ValueError("Head-tail requires room for at least two content tokens")
            head = (budget + 1) // 2
            for start in range(0, len(documents), 256):
                items = tokenizer(
                    documents[start:start + 256], add_special_tokens=False,
                    truncation=False, verbose=False,
                )
                for ids in items.input_ids:
                    if len(ids) <= length - tokenizer.num_special_tokens_to_add(pair=False):
                        final_ids = tokenizer.build_inputs_with_special_tokens(ids)
                    else:
                        final_ids = tokenizer.build_inputs_with_special_tokens(
                            ids[:head], ids[-(budget - head):]
                        )
                    self.records.append({"input_ids": final_ids, "attention_mask": [1] * len(final_ids)})
        if labeled:
            for record, label in zip(self.records, frame.label):
                record["labels"] = LABELS.index(label)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if class_weights is not None:
            self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.class_weights is None:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()
        loss = torch.nn.functional.cross_entropy(
            logits, labels, weight=self.class_weights.to(logits.device)
        )
        return (loss, outputs) if return_outputs else loss


def run(args):
    text, _, val, fit = partitions(args)
    test = load(args.data / "test.csv", ["id", text])
    args.out.mkdir(parents=True, exist_ok=False)
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=args.revision, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id={label: i for i, label in enumerate(LABELS)},
    )
    if args.length > tokenizer.model_max_length:
        raise ValueError(f"Tokenizer supports at most {tokenizer.model_max_length} tokens")
    fit_data = EncodedDocuments(fit, text, tokenizer, args.length, args.strategy, True)
    val_data = EncodedDocuments(val, text, tokenizer, args.length, args.strategy, True)
    test_data = EncodedDocuments(test, text, tokenizer, args.length, args.strategy, False)
    weights = None
    if args.weighted:
        counts = fit.label.value_counts().reindex(LABELS).to_numpy()
        weights = torch.tensor(len(fit) / (len(LABELS) * counts), dtype=torch.float32)

    def metrics(prediction):
        labels = np.array(LABELS)[prediction.label_ids]
        return {"macro_f1": scores(labels, softmax(prediction.predictions, axis=1))["macro_f1"]}

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    training = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        eval_strategy="epoch", save_strategy="epoch", logging_steps=100,
        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accumulation, weight_decay=.01, warmup_ratio=.06,
        seed=SEED, data_seed=SEED, bf16=bf16, fp16=cuda and not bf16,
        report_to="none", disable_tqdm=True, save_only_model=True,
    )
    trainer = WeightedTrainer(
        model=model, args=training, train_dataset=fit_data, eval_dataset=val_data,
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics, class_weights=weights,
    )
    trainer.train()
    best_checkpoint = trainer.state.best_model_checkpoint
    val_prob = softmax(trainer.predict(val_data).predictions, axis=1)
    test_prob = softmax(trainer.predict(test_data).predictions, axis=1)
    result = write_predictions(args.out, val, val_prob, test, test_prob)
    save_metrics(args, {"encoder": result}, best_checkpoint=best_checkpoint,
                 model_commit=getattr(model.config, "_commit_hash", None),
                 epoch_history=trainer.state.log_history)
