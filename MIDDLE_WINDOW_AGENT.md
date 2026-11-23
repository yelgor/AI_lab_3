# Експеримент на готовому XLM-R

На Linux-ПК користувача онови гілку homework-3-pipeline. Використай
попередній Python із CUDA, офіційні CSV, той самий split.json і
checkpoint-1875. Попередній ансамбль має 0.89542 validation macro-F1
і 0.85822 F1 для fusion. Збережи його CSV.

Підстав фактичні шляхи. REF має вказувати на val.csv і test.csv від
того самого чекпойнта зі стратегією head-tail і довжиною 384 токени.
Запуск обчислює середнє вікно лише для документів довших за ліміт.
У коротких документах оцінки беруться з REF. Ваги моделі не оновлюються.

```bash
PY=.venv/bin/python
DATA=data
RUNS=.local-training/runs
SPLIT="$RUNS/split.json"
REF="$RUNS/xlmr-epoch1-probabilities"
CHECKPOINT="$RUNS/xlmr-60k-head-tail/checkpoints/checkpoint-1875"
MIDDLE="$RUNS/xlmr-middle"
CPU="$RUNS/hash-full-distributed"

"$PY" score_middle.py --data "$DATA" --split "$SPLIT" \
  --checkpoint "$CHECKPOINT" --reference "$REF" --out "$MIDDLE"
```

Якщо CUDA не вистачає пам’яті, спробуй --batch 4. У metrics.json
порівняй macro-F1 і F1 для fusion на всій валідації та окремо на
довгих документах. Потім перевір ансамблі з CPU:

```bash
"$PY" lab.py --data "$DATA" --split "$SPLIT" select \
  "$REF" "$CPU" "$MIDDLE/middle_weight_0.1" \
  "$MIDDLE/middle_weight_0.2" "$MIDDLE/middle_weight_0.3" \
  --all-pairs --out "$RUNS/select-middle"
```

Повідом час, метрики для довгих текстів, обрані ваги і шлях до CSV.
Порівняй фінальну оцінку з 0.89542 на тій самій валідації. Якщо
приросту немає, збережи попередній CSV як основний. Якщо довгі тексти
помітно виграють, це аргумент на користь окремого експерименту з
навчанням багатовіконної моделі за міткою документа. Публічний
Kaggle score нового CSV стане відомим після подання.
