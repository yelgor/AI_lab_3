# Локальний запуск на ПК з RTX 4060 (8 ГіБ VRAM)

Працюй у гілці `homework-3-pipeline` з Python 3.11 або 3.12. Далі команди для PowerShell. Файли `train.csv`, `test.csv` і `sample_submission.csv` поклади в `data/`. Для іншої папки передавай `--data ШЛЯХ` перед підкомандою.

## 1. Підготувати спліт

```powershell
git fetch origin
git switch homework-3-pipeline
git pull --ff-only
python -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r requirements.txt
```

Якщо `runs/split.json` уже є, використай його. Інакше один раз виконай `& $py lab.py prepare`. Якщо в CSV є поле з ID оригінального документа, додай до цієї команди `--group НАЗВА_ПОЛЯ`. Усі порівняння мають використовувати цей спліт. Для EDA виконай `& $py lab.py eda`, додавши `--language` і `--domain` з фактичними назвами колонок, якщо вони є.

## 2. CPU baseline з низьким споживанням RAM

Перший запуск використовує **8 000 навчальних текстів, до 700 символів на документ для TF-IDF і не більше 20 000 ознак на векторизатор**. Останній ліміт застосовується після збору кандидатів, тому для RAM важливі також перші два. Валідація і всі 25 000 тестових рядків залишаються повними. Стилометрія читає текст цілком.

Збережи цей файл як `local_baseline.py` у корені репозиторію:

```python
import argparse

from sklearn.model_selection import train_test_split

import lab

original_partitions = lab.partitions
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--fit-rows", type=int, default=8000)
limits, remaining = parser.parse_known_args()


def subset_partitions(args):
    text, _, val, sample = original_partitions(args)
    if len(sample) > limits.fit_rows:
        sample, _ = train_test_split(
            sample,
            train_size=limits.fit_rows,
            stratify=sample.label,
            random_state=lab.SEED,
        )
    print(f"Baseline fit rows: {len(sample)}", flush=True)
    return text, sample, val, sample


if __name__ == "__main__":
    import sys

    sys.argv[1:] = remaining
    lab.partitions = subset_partitions
    lab.main()
```

Запускай навчання одним процесом. Закрий важкі програми й стеж за пам’яттю в Task Manager. У Linux можна обгорнути команду в `/usr/bin/time -v` для вимірювання піку RSS.

```powershell
& $py local_baseline.py baseline --fit-rows 8000 --features 20000 --max-chars 700 --c 0.25 1 --out runs/local-cpu-8k
& $py lab.py select runs/local-cpu-8k/tfidf runs/local-cpu-8k/stylometry --out runs/local-first
```

Перший сабміт з'явиться в `runs/local-first/submission_1.csv`. Пік RAM залежить від текстів і доступної системної пам’яті. Якщо пам’ять наближається до межі, зупини процес і повтори в новий `--out` з `--fit-rows 4000 --max-chars 400 --features 12000`. Якщо перший запуск стабільний і є виміряний запас, спробуй 15 000 текстів і 1 000 символів в окремому каталозі. Збережи команду, пік RAM, час і `metrics.json` для кожного запуску. У Bash заміни `& $py` на `.venv/bin/python`.

## 3. GPU після готового CPU сабміту

Установи CUDA-збірку PyTorch у те саме оточення за командою з [офіційного селектора](https://pytorch.org/get-started/locally/). Потім:

```powershell
& $py -m pip install -r requirements-gpu.txt
& $py -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
& $py lab.py encoder --batch 1 --accumulation 32 --length 192 --epochs 1 --out runs/local-head
```

GPU запускай після завершення CPU, щоб процеси не ділили системну RAM. Якщо навіть batch 1 не вміщується у VRAM, спробуй `--length 128` в окремому каталозі. Якщо GPU запуск пройшов, за наявності часу повтори ті самі параметри з `--strategy head-tail` або `--weighted`, кожен в окремому каталозі. У фінальний відбір додай лише завершені запуски.

```powershell
& $py lab.py select runs/local-cpu-8k/tfidf runs/local-cpu-8k/stylometry runs/local-head --out runs/local-final
```

Вибір робиться за validation macro-F1. Запиши також F1 для `fusion` і матрицю помилок. Для іншого набору моделей використовуй новий `--out`. Сабміть CSV вручну. Експерименти й фактичні метрики передай студенту для власного звіту.

## 4. Збільшити вибірку після першого сабміту

Спочатку збережи попередні CSV і метрики. Використовуй той самий
`runs/split.json`. Новий скрипт обчислює IDF лише на навчальних рядках,
хешує n-грами у фіксований простір ознак і обробляє розріджені матриці
пакетами. Вхідні CSV усе ще завантажуються в RAM. Запускай по одній команді
та перевір пікове споживання пам'яті перед наступною.

```powershell
& $py large_baseline.py --fit-rows 60000 --max-chars 2000 --batch 256 --epochs 2 --out runs/hash-60k
& $py lab.py select runs/local-cpu-8k/tfidf runs/local-cpu-8k/stylometry runs/hash-60k --out runs/select-60k
```

Якщо запуск стабільний і є час, збільш вибірку зі збереженням решти
параметрів. Якщо пам'яті бракує під час векторизації, повтори з
`--batch 128` і новим вихідним каталогом.

```powershell
& $py large_baseline.py --fit-rows 120000 --max-chars 2000 --batch 256 --epochs 2 --out runs/hash-120k
& $py lab.py select runs/local-cpu-8k/tfidf runs/local-cpu-8k/stylometry runs/hash-60k runs/hash-120k --out runs/select-120k
```

Підстав фактичні шляхи до першого baseline, якщо вони відрізняються.
Якщо вже є завершений XLM-R, додай його каталог у `select`. Порівнюй
macro-F1 і F1 для fusion на спільній валідації. Публічний результат
0.68966 належить попередньому сабміту. Новий публічний результат
стане відомий після подання нового CSV.
