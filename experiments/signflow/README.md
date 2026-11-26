# Контрольный прогон распознавания РЖЯ

## Текущая сборка со Slovo

Приложение выбирает самое вероятное слово из `app/src/main/assets/phrases/catalog.json` по оценкам готовой модели [Slovo `mvit32-2.onnx`](https://github.com/hukenovs/slovo). При открытии распознавания оно копирует веса из APK, если файл есть в `app/src/main/assets/recognition/`, или загружает их с сервера авторов.

В Android Studio выберите конфигурацию `app` и нажмите **Run**. В режиме «РЖЯ → текст» разрешите доступ к камере, начните захват и покажите жест. Нажмите «Стоп» примерно через секунду после жеста. Последняя секунда записи исключается при подготовке 32 кадров.

### Контрольный клип на Mac и Android

Для локальной подготовки тензоров нужны FFmpeg, JDK и [ONNX Runtime Java 1.23.2](https://repo.maven.apache.org/maven2/com/microsoft/onnxruntime/onnxruntime/1.23.2/onnxruntime-1.23.2.jar). Задайте каталог контрольных файлов и ожидаемую метку клипа.

```sh
control=/absolute/path/to/control
mkdir -p "$control"
printf '%s\n' 'Привет' > "$control/mvit-expected.txt"
python3 experiments/signflow/control.py \
  --ort-jar /path/to/onnxruntime-1.23.2.jar \
  --model /path/to/mvit32-2.onnx \
  --manifest app/src/main/assets/recognition/manifest.json \
  --clip /path/to/hello.mp4 \
  --save-tensor "$control/mvit-control.f32" \
  --save-scores "$control/mvit-scores.json" --expected 'Привет'

./gradlew :app:connectedDebugAndroidTest -PsignflowControlDir="$control"
```

Android сравнивает метку и оценки всех 1001 классов на том же входном тензоре. Для 50 проверок из `SignFlowGestureTest` сохраните через `control.py --save-tensor` два тензора каждого жеста по путям `mvit-gestures/<адрес-карточки>/1.f32` и `mvit-gestures/<адрес-карточки>/2.f32` в том же каталоге.

## Архивная проверка SignFlow‑R

Конфигурация прежнего эксперимента находится в `legacy/`. Для его воспроизведения положите свои веса `SignFlow-R.onnx` в эту папку. Результаты обучения приведены в [отчёте](household-training.md).

Контрольное видео [«Привет» из Slovo](https://github.com/hukenovs/slovo/blob/main/examples/f17a6060-6ced-4bd1-9886-8578cfbb864f.mp4) содержит 112 кадров, 720 × 1280, 30 кадр/с. Ожидаемая метка указана в [таблице аннотаций](https://sp.susu.ru/student/masterthesis/2024_229_sviridovma.pdf). SHA‑256 MP4 — `98da3c5da34c473e5c1909db66c1fc81ce694f9ff59db1d392920ecf7bcf17f4`.

В [репозитории Slovo](https://github.com/hukenovs/slovo#signflow-models) для SignFlow‑R заявлено 267 классов. Публичная демонстрация ссылается на `mvit32-2.onnx`. Для контрольного прогона нужны веса SignFlow‑R, порядок 267 классов для этого файла и исходный код или спецификация препроцессинга, включая выбор кадров, цвет, ориентацию, обрезку, масштабирование и нормализацию. При получении комплекта также проверьте, что «Привет» входит в словарь модели.

### Проверка на Mac

Установите FFmpeg и используйте JDK и [ONNX Runtime Java 1.23.2](https://repo.maven.apache.org/maven2/com/microsoft/onnxruntime/onnxruntime/1.23.2/onnxruntime-1.23.2.jar). Команда без `--clip` печатает входы, выходы, типы, размеры и метаданные ONNX.

```sh
python3 experiments/signflow/control.py \
  --java "/Applications/Android Studio.app/Contents/jbr/Contents/Home/bin/java" \
  --ort-jar /path/to/onnxruntime-1.23.2.jar \
  --model /path/to/SignFlow-R.onnx
```

Для исторического прогона использовался рецепт `rgb-center-crop-bilinear-uniform-v1`. Он масштабирует кадры с сохранением пропорций, берёт центральную область и выбирает ближайшие кадры к равномерным временным отметкам.

Файлы в каталоге `recognition`:

| Файл | Содержимое |
| --- | --- |
| `SignFlow-R.onnx` | Проверенные веса |
| `labels.json` | Массив из 267 непустых строк в порядке выхода модели |
| `manifest.json` | Поля `recipe`, `frames`, `width`, `height`, `layout` (`N1CTHW`, `NCTHW` или `NTCHW`), тройки `mean` и `std` в шкале 0–255, `modelSha256` и `labelsSha256` |

Подготовьте отдельный каталог для контроля, поместите комплект в его подкаталог `recognition` и выполните:

```sh
control=/absolute/path/to/control
python3 experiments/signflow/control.py \
  --java "/Applications/Android Studio.app/Contents/jbr/Contents/Home/bin/java" \
  --ort-jar /path/to/onnxruntime-1.23.2.jar \
  --model "$control/recognition/SignFlow-R.onnx" \
  --manifest "$control/recognition/manifest.json" \
  --labels "$control/recognition/labels.json" \
  --clip /path/to/f17a6060-6ced-4bd1-9886-8578cfbb864f.mp4 \
  --save-tensor "$control/control.f32" --expected 'Привет'
```

### Результат проверки

- На Mac декодированы 112 кадров контрольного MP4. Для проверки цепочки использован временный диагностический граф с входом `[1, 3, 8, 32, 32]` и выходом `[1, 267]`. Получен ожидаемый тестовый индекс 66.
- На ARM64-эмуляторе `SignFlowControlTest` прошёл с тем же тензором. С временным графом также проверены превью CameraX, захват, появление кандидата, подтверждение и повтор.
- Скачанный `SignFlow-R.onnx` открылся в ONNX Runtime 1.23.2 на Mac. SHA‑256 — `49c0d4023a64188b2cdd9a31e00d610efa9d15902005bf9f319c61c2a0808f65`. Вход `input` — float32 `[1, 1, 3, 32, 224, 224]`. Выход `output` — float32 `[1, 267]`. Метаданные PyTorch-экспорта не содержат карту классов или рецепт подготовки кадров.
- На ролике «Привет» экспериментальный рецепт дал максимальную оценку 0,6352845 на индексе 164. Взяты равномерно 32 RGB-кадра, применены билинейное масштабирование с центральной обрезкой до 224 × 224 и нормализация `mean=[123.675,116.28,103.53]`, `std=[58.395,57.12,57.375]` из конфига демонстрации Slovo. Эти параметры и связь индекса 164 с русской меткой требуют подтверждения для SignFlow‑R.
- Тест на ARM64-эмуляторе подтвердил совпадение всех 267 оценок Mac и Android в пределах 0,001 для этого же тензора и тех же весов. Тест русской метки пропущен до получения проверенной карты классов.

Тренер принимает согласованный словарь через
`--catalog experiments/signflow/legacy/catalog.json`. Для обучения нужны размеченные
прогоны SignFlow‑R для всех слов и для класса `__other__`. Обучающие записи и
контрольные записи следует разделить по исполнителям. Метрика `appTop1` показывает,
как часто на тестовых записях известных жестов модель выбирает верное слово по
правилу Android-приложения.

### Прямые привязки по роликам Slovo

Скрипт выборочно читает архив Slovo по ссылке авторов и извлекает записи только
для слов каталога приложения. Если архив уже скачан, передайте путь через
`--archive`. В отчёте видны исходные исполнители и слова без записей.

```sh
python3 experiments/signflow/extract_slovo.py --output /tmp/slovo-catalog
python3 experiments/signflow/map_catalog.py --videos /tmp/slovo-catalog \
  --catalog app/src/main/assets/phrases/catalog.json \
  --ort-jar /absolute/path/to/onnxruntime.jar \
  --output /tmp/signflow-candidate-manifest.json
```

Второй скрипт запускает архивный SignFlow‑R и готовит экспериментальный манифест
с найденными привязками. Он отмечает конфликты индексов и расхождения между
роликами.
