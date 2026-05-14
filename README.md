# Экстрактор маски кожных покровов лица

## Описание

Проект выделяет кожные покровы лица на изображении с использованием `dlib` и модели `68 facial landmarks`.
Алгоритм строит полигон лица по опорным точкам, экстраполирует область лба вверх, затем вырезает из маски области глаз и рта.
Результатом является изображение, где сохранены только пиксели кожи лица.

Ключевые особенности:
- поддержка обработки одного файла и пакетной обработки директории;
- C++-friendly data flow с явными буферами и типизацией `numpy` массивов;
- переиспользование загруженного детектора landmarks в batch-режиме для снижения накладных расходов.

## Подробности алгоритма
Детальное описание математического аппарата, метода экстраполяции лба и логики исключения глаз/рта вынесено в отдельный технический документ: [METHODOLOGY.md](METHODOLOGY.md).

## Установка

### 1) Установите зависимости

```bash
pip install -r requirements.txt
```

### 2) Подготовьте веса модели dlib

Необходимо скачать файл `shape_predictor_68_face_landmarks.dat` и разместить его в корне проекта.

Команды для загрузки:

- Windows (PowerShell):

```powershell
Invoke-WebRequest -Uri "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -OutFile "shape_predictor_68_face_landmarks.dat.bz2"
tar -xjf "shape_predictor_68_face_landmarks.dat.bz2"
Remove-Item "shape_predictor_68_face_landmarks.dat.bz2"
```

- Linux/macOS (bash):

```bash
curl -L -o shape_predictor_68_face_landmarks.dat.bz2 "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

По умолчанию скрипт ожидает путь:

```text
shape_predictor_68_face_landmarks.dat
```

При необходимости можно передать свой путь через аргумент `--predictor`.

## Использование

Точка входа:

```bash
python solution.py [аргументы]
```

По умолчанию:
- вход: `data/input`
- выход: `data/output`

### Режим 1: обработка одного файла (Single-file mode)

```bash
python solution.py --input "data/input/input_img_1.jpg" --output "data/output/out_1.jpg"
```

Также можно передать выход как директорию, тогда имя файла сохранится автоматически:

```bash
python solution.py --input "data/input/input_img_1.jpg" --output "data/output"
```

### Режим 2: пакетная обработка директории (Batch mode)

Если `--input` указывает на директорию, обрабатываются все файлы с расширениями:
`.jpg`, `.jpeg`, `.png`.

```bash
python solution.py --input "data/input" --output "data/output"
```

Или просто с дефолтными путями:

```bash
python solution.py
```

Результаты сохраняются в `data/output` с оригинальными именами файлов.
Во время обработки выводится прогресс, например: `Processed 5/10 images`.

## Аргументы CLI

- `--input` - путь к входному файлу или директории (по умолчанию `data/input`);
- `--output` - путь к выходному файлу или директории (по умолчанию `data/output`);
- `--predictor` - путь к файлу `shape_predictor_68_face_landmarks.dat`;
- `--forehead-offset-ratio` - коэффициент экстраполяции лба (по умолчанию `0.25`);
- `--allow-empty-face` - не падать, если лицо не найдено.

## Структура проекта

- `solution.py` - CLI-точка входа, парсинг аргументов, запуск single/batch обработки.
- `pipeline.py` - оркестратор пайплайна, маршрутизация file/dir режимов, прогресс-логирование.
- `landmarks_dlib.py` - адаптер dlib, извлечение landmarks в формат `(N, 68, 2) int32`.
- `skin_mask_math.py` - чистые функции построения маски кожи (лоб + вычитание глаз/рта).
- `image_io.py` - загрузка и сохранение изображений.
- `cv_types.py` - типы и валидация буферов (`dtype`, `shape`, contiguous).
- `tests/` - unit/smoke тесты алгоритма и пайплайна.

## Примечания по производительности

- В batch-режиме детектор (`DlibLandmarkDetector`) создается один раз и переиспользуется.
- В алгоритме используются `out`-буферы там, где это возможно, чтобы снизить количество лишних аллокаций.
- Контракты массивов (тип/размер/непрерывность памяти) валидируются для предсказуемого поведения и упрощения будущего C/C++ портирования.
