# Лабораторная работа 1 (CV)

## Тема
Проведение исследований моделями обнаружения и распознавания объектов с использованием Ultralytics (семейство YOLOv11).

## 1. Выбор начальных условий

### 1.1 Выбранный датасет и обоснование
- Датасет: Hard Hat Workers (Roboflow Universe).
- Ссылка: https://universe.roboflow.com/joseph-nelson/hard-hat-workers
- Используемая версия в этом репозитории: v8 (raw_HelmetClassOnly).
- Практическая задача: автоматический контроль соблюдения требований техники безопасности в производственных/строительных зонах по видео с камер.

Почему это реальная задача:
- снижает риск травм за счет раннего обнаружения нарушений;
- автоматизирует рутинный мониторинг зон с повышенной опасностью;
- может использоваться как модуль в CPS/SCADA-процессе для сигнализации оператору.

Важно по текущей версии данных:
- в версии v8 в разметке один класс `helmet` (nc=1);
- это позволяет напрямую решать задачу детекции касок для базового контроля СИЗ.

### 1.2 Выбранные метрики и обоснование
Основные метрики качества для object detection:
- mAP50-95 (главная интегральная метрика качества локализации и детекции);
- mAP50 (более интерпретируемая метрика для сравнения экспериментов);
- Precision (минимизация ложных срабатываний);
- Recall (минимизация пропусков объектов);

Почему именно они:
- mAP50-95 является стандартом сравнения детекторов и учитывает разные пороги IoU;
- Precision важна, чтобы не создавать избыточные тревоги;
- Recall важна для безопасности, где пропуск объекта критичнее;
- вместе метрики дают баланс между надежностью оповещения и стабильностью системы.

## 2. Реализация

В репозитории реализован единый скрипт:
- `src/lab1_yolo11.py`

Поддерживаются режимы:
- `train` - обучение модели;
- `eval` - оценка сохраненных весов на val/test;
- `predict` - инференс на изображениях/папке;
- `experiments` - серия экспериментов (например, YOLO11n и YOLO11s).

Также добавлен локальный конфиг датасета:
- `dataset/data.local.yaml`

## 3. Воспроизводимая инструкция установки и запуска

Ниже шаги, которые можно выполнить с нуля.

### 3.1 Требования
- Рекомендуется Python 3.12 (на практике наиболее стабильно для PyTorch + CUDA)
- Windows/Linux/macOS

### 3.2 Установка

1. Клонировать репозиторий и перейти в папку проекта.
2. Создать виртуальное окружение и активировать его.
3. Установить зависимости:

```bash
pip install -r requirements.txt
```

### 3.2.1 (Опционально) Включение GPU на Windows + NVIDIA

Если после установки из `requirements.txt` PyTorch оказался CPU-only, переустановите его CUDA-сборку:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio
```

Проверка, что CUDA доступна:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 3.3 Проверка структуры данных

Ожидается структура:

```text
dataset/
  data.local.yaml
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels
```

### 3.4 Обучение базовой модели (YOLO11n)

```bash
python src/lab1_yolo11.py train --model yolo11n.pt --epochs 30 --device cpu
```

Вариант для GPU (рекомендуется):

```bash
python src/lab1_yolo11.py train --model yolo11n.pt --epochs 30 --device 0 --workers 4 --name helmet_detector_gpu
```

Результаты будут сохранены в папку:
- `runs/train/helmet_detector`

Внутри будет файл:
- `metrics_summary.json` (сводка метрик на test split)

### 3.5 Запуск серии экспериментов

```bash
python src/lab1_yolo11.py experiments --models n_s --epochs 20 --device cpu
```

Что делает команда:
- обучает `yolo11n.pt` и `yolo11s.pt`;
- сохраняет результаты в `runs/experiments/...`;
- формирует `metrics_summary.json` для каждого запуска.

### 3.6 Оценка произвольных весов

```bash
python src/lab1_yolo11.py eval --weights runs/train/helmet_detector/weights/best.pt --split test
```

По умолчанию метрики сохраняются в:
- `runs/eval/metrics_test.json`

### 3.7 Инференс на тестовых изображениях

```bash
python src/lab1_yolo11.py predict --weights runs/train/helmet_detector/weights/best.pt --source dataset/test/images --device cpu
```

Предсказания будут сохранены в:
- `runs/predict/helmet_detector`
