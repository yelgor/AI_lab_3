# Homework 3 · результати локального запуску

Фінальний ансамбль отримав **0,89542 macro-F1 на локальній валідації**
та **0,90867 на Kaggle за результатом, повідомленим після подання**.
Усі локальні порівняння використовували той самий спліт із 217 881
навчальним і 54 471 валідаційним текстом. Фінальний CSV містить 25 000
прогнозів у порядку ID з `sample_submission.csv`.

| Модель | Навчальні тексти | Validation macro-F1 | F1 класу fusion |
| --- | ---: | ---: | ---: |
| Хешований TF-IDF + SGD, head | 120 000 | 0,76071 | 0,67114 |
| Хешований TF-IDF + SGD, head | 217 881 | 0,76098 | 0,66902 |
| Хешований TF-IDF + SGD, distributed | 217 881 | 0,85484 | 0,80118 |
| XLM-R head-tail, чекпойнт першої епохи | 60 000 | 0,87900 | 0,83699 |
| Ансамбль | | **0,89542** | **0,85822** |

Ансамбль усереднює ймовірності XLM-R і повної моделі `distributed` з
вагами 0,5 і 0,5, а потім множить оцінку класу `fusion` на `exp(-0.1)`
перед вибором мітки. Відбір ваг і корекції використовував ту саму
валідацію, на якій наведено macro-F1. Результат XLM-R у таблиці отримано
окремим прогоном із чекпойнта. Лог навчання після першої епохи показував
0,87918. Окремий прогін використовував більший пакет для інференсу.

Повний CPU-запуск `distributed` тривав 36 хв 10 с і мав пік RSS 2,09 ГіБ.
Повний `head` тривав 34 хв 25 с із піком RSS 2,08 ГіБ. Прогнози з
чекпойнта XLM-R на RTX 4060 тривали 7 хв 44 с. Збільшення вибірки для
`head` дало лише 0,027 відсоткового пункта macro-F1. Запуск
`distributed` також відрізнявся лімітом символів і кількістю епох.

## Відтворення прогнозів і відбору

Використай оригінальний спліт, офіційні CSV і чекпойнт XLM-R першої
епохи (`checkpoint-1875`). Параметри навчання є в `LOCAL_AGENT.md`
та `FULL_CPU_AGENT.md`. Потрібне середовище з `requirements-gpu.txt`
і робочою CUDA. Підстав шляхи до готових каталогів моделей.

```bash
PY=.venv/bin/python
DATA=data
RUNS=runs
SPLIT="$RUNS/split.json"
CHECKPOINT="$RUNS/xlmr-60k-head-tail/checkpoints/checkpoint-1875"
XLMR="$RUNS/xlmr-epoch1-probabilities"

"$PY" score_epoch1.py --data "$DATA" --split "$SPLIT" \
  --checkpoint "$CHECKPOINT" --device cuda --out "$XLMR"
"$PY" lab.py --data "$DATA" --split "$SPLIT" select \
  "$XLMR" "$RUNS/hash-120k-cache" "$RUNS/hash-full-head" \
  "$RUNS/hash-full-distributed" "$RUNS/hash-60k" \
  "$RUNS/local-cpu-8k/tfidf" "$RUNS/local-cpu-8k/stylometry" \
  --all-pairs --out "$RUNS/select-full-cpu-xlmr"
```

Новий CSV з'явиться у `runs/select-full-cpu-xlmr/submission_1.csv`.
Дані, чекпойнти й локальні артефакти зберігаються окремо від Git.
