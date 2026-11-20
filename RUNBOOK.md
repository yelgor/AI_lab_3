# Homework 3 experiment runbook

Use Python 3.11 or newer. Run commands from the repository root. Put the three
official competition files in `data/` or pass `--data PATH` before the
subcommand. Install the CPU dependencies with `python -m pip install -r
requirements.txt`. For encoder runs, enable a GPU in Colab or use local CUDA.
Install `requirements-gpu.txt` in that environment. In Colab, retain the
dataset, split, and run artifacts on persistent storage. Do not commit them.

## Instructions for the local runner

1. Inspect only the CSV column names to identify the text, language, domain,
   and original document ID columns. Use the grouping argument only when a
   column actually identifies versions of the same source document. The
   splitter also groups identical text after case and whitespace normalization.
   If a source ID is absent, record that paraphrased duplicates can still cross
   the split. Use metadata for EDA and splitting. Fit models from text alone.
2. Run the commands below in sequence. Replace the optional column arguments
   with actual column names when available. Keep the manifest and the encoder
   sample fixed. Record Python and package versions, GPU, commands, runtime,
   and failures. Store distinct runs in distinct output directories. On CUDA
   out-of-memory, use a smaller batch and proportionally larger accumulation
   for both members of the comparison, then rerun the pair.
3. Rank runs by validation macro-F1. Read the per-class precision, recall, F1,
   and the raw confusion matrix in each metrics JSON. Compare stylometry with
   the selected TF-IDF baseline. Compare head-tail against head and weighted
   head against unweighted head. The TF-IDF baseline trains on the full fit
   partition, while encoder variants train on the same fixed subset. Compare
   all variants on the full common validation partition. A leaderboard score
   is an external result, so choose parameters and blending weights internally.
4. Keep a log of hypotheses, actual results, and failures. Present the log and
   artifacts to the student for their own report and defense. The local agent
   can manage jobs and parse metrics. Train using only official training data
   and allowed open weights. Keep raw competition text out of proprietary AI
   services and use no external corpus or test labels.
5. Review the validation metrics before manually submitting the generated file
   to Kaggle. The notebook environment can run the same commands with `!`.
   Keep the artifacts associated with any selected Kaggle submission.

```bash
python lab.py prepare
# Example when metadata exists:
# python lab.py prepare --text text --group source_id
python lab.py eda
# Optional: python lab.py eda --language language --domain domain
python lab.py baseline
python lab.py select runs/baseline/tfidf runs/baseline/stylometry --out runs/first
python lab.py encoder --out runs/head
python lab.py encoder --strategy head-tail --out runs/head-tail
python lab.py encoder --weighted --out runs/head-weighted
python lab.py select runs/baseline/tfidf runs/baseline/stylometry runs/head runs/head-tail runs/head-weighted
```

If the CPU baseline runs out of memory, try `--features 60000 --max-chars 4000`
with a fresh `--out` directory. Once the core experiments finish, extra budget
can go to a paired 384-token head/head-tail comparison. Keep the training
sample and validation IDs fixed, and change only the setting being tested.
To rerun an encoder with the same pretrained snapshot, pass its logged
`model_commit` as `--revision`.

`select` aligns rows by ID and tries single models, an equal-weight blend,
three blends of the two best single models, and four fusion score adjustments
on the strongest candidate. Softmax of SVM margins provides normalized scores
for blending. Reusing validation for model selection gives an optimistic
estimate of private leaderboard performance. The selected file
`runs/final/submission_1.csv` follows sample order and checks for 25,000 test
IDs. Only use `--expected` to smoke-test on a synthetic fixture.

The baseline exports the largest positive coefficients by class for optional
explainability. A coefficient describes a learned association. Look at which
features are present in a specific document before attributing its prediction.
EDA writes short examples and distribution tables. Use the actual text and
metadata in those files to investigate when the classes differ.
