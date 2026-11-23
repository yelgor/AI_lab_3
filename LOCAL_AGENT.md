# Завдання для локального агента

Ти працюєш на Linux-ПК користувача з RTX 4060 (8 ГіБ VRAM). Мета — згенерувати CSV для нових текстів і порівняти його з уже готовим варіантом за macro-F1 на фіксованій валідації. Орієнтири з попередніх запусків: 0,75578 (hash-60k), 0,76071 (hash-120k), 0,76483 (відбір).

## Підготовка

1. У локальному клоні перевір `git status --short` і гілку. Онови `homework-3-pipeline` через `git pull --ff-only`. Збережи локальні зміни та попередні артефакти. Якщо гілка розійшлася, використай окремий worktree з віддаленої гілки й абсолютні шляхи до даних, спліту та результатів.
2. Знайди в попередній команді або `.local-training/training.log` фактичні шляхи до Python, офіційних CSV і попереднього `split.json`. Використай цей самий спліт. Перевір, що старі моделі мають `val.csv` і `test.csv`, а попередній фінальний CSV доступний у `.local-training/runs/select-120k-cache/`.
3. Перевір вільну RAM і місце на диску. Запускай по одному процесу. Пиши час, пік RSS і помилки в лог. У Linux для цього можна обгорнути команду в `/usr/bin/time -v`. Вихідні каталоги для нових запусків мають бути новими.

```bash
# Задай фактичні шляхи. PY має вказувати на Python чинного оточення.
PY=python
DATA="$PWD/data"
RUNS="$PWD/.local-training/runs"
SPLIT="$RUNS/split.json"
test -f "$DATA/train.csv" &&
test -f "$DATA/test.csv" &&
test -f "$DATA/sample_submission.csv" &&
test -f "$SPLIT"
```

Перевір код виходу цієї перевірки перед навчанням. Якщо дедлайн близько, повідом користувачеві, де лежить попередній готовий CSV.

## Запуски

Спочатку навчи пакетний TF-IDF на всій навчальній частині. За нестачі пам’яті під час векторизації повтори з `--batch 128` і новим каталогом.

```bash
/usr/bin/time -v "$PY" large_baseline.py --data "$DATA" --split "$SPLIT" \
  --fit-rows 300000 --coverage distributed --max-chars 4000 \
  --batch 256 --epochs 3 --out "$RUNS/hash-full-distributed"
```

Перед XLM-R перевір CUDA в тому самому оточенні. Якщо залежностей бракує, встанови CUDA-збірку PyTorch і `requirements-gpu.txt`.

```bash
"$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
/usr/bin/time -v "$PY" lab.py --data "$DATA" --split "$SPLIT" encoder \
  --fit-rows 60000 --strategy head-tail --length 384 --batch 1 \
  --accumulation 32 --epochs 3 --gradient-checkpointing \
  --out "$RUNS/xlmr-60k-head-tail"
```

У разі CUDA OOM повтори з `--length 256` у новому каталозі. Якщо є час на контрольний запуск, повтори успішні параметри з `--strategy head`, змінивши тільки стратегію та вихідний каталог.

## Відбір і фінальний статус

Щойно завершиться новий CPU-запуск, можна виконати відбір з ним і старими моделями. Після завершення GPU виконай відбір ще раз у новому каталозі. Підстав наявні каталоги моделей із `val.csv` і `test.csv`. Каталог попереднього фінального CSV збережи для прямого порівняння метрик.

```bash
"$PY" lab.py --data "$DATA" --split "$SPLIT" select \
  "$RUNS/hash-60k" "$RUNS/hash-120k-cache" \
  "$RUNS/hash-full-distributed" "$RUNS/xlmr-60k-head-tail" \
  --all-pairs --out "$RUNS/select-strong"
```

Повідом користувача про завершені й невдалі запуски, macro-F1, F1 класу fusion, час, пік RAM і шлях до `submission_1.csv`. Рекомендуй новий файл за підсумком валідації. Якщо відбір не перевищив 0,76483, збережи попередній CSV як основний. Подання на Kaggle залиш користувачеві. Для навчання використовуй тільки офіційні навчальні дані й дозволені відкриті ваги.
