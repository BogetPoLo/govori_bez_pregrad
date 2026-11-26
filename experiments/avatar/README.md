# Анимации РЖЯ для Android

Архивный эксперимент. Текущий экран «Русский текст → РЖЯ» открывает карточки видеословаря SignFlow. Ниже описана подготовка аватара из предыдущей версии прототипа.

## Подготовка

Понадобятся Python 3.12, FFmpeg и локальные веса. MediaPipe оценивает позу тела, 21 точку каждой кисти и ориентиры лица. Модели тела и рук запускаются через OpenCV DNN, сетка лица через LiteRT. Фронтальный кадр одинаков для всего ролика. Сглаживание тела и запястья использует пять кадров, локальная форма пальцев три кадра. Для источника в 30 кадр/с добавляется промежуточный кадр между каждой парой исходных, результат сохраняется в 60 кадр/с.

```sh
python3.12 -m venv /tmp/signflow-avatar-venv
. /tmp/signflow-avatar-venv/bin/activate
python -m pip install opencv-contrib-python==4.12.0.88 ai-edge-litert==2.3.0 Pillow==13.0.0
```

Из [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models) положите в одну папку следующие веса и одноимённые обёртки из каталогов моделей:

| Веса | Обёртка |
| --- | --- |
| `person_detection_mediapipe_2023mar.onnx` | `mp_persondet.py` |
| `pose_estimation_mediapipe_2023mar.onnx` | `mp_pose.py` |
| `palm_detection_mediapipe_2023feb.onnx` | `mp_palmdet.py` |
| `handpose_estimation_mediapipe_2023feb.onnx` | `mp_handpose.py` |
| `face_detection_yunet_2023mar.onnx` | `FaceDetectorYN` из OpenCV |

В ту же папку извлеките `face_landmarks_detector.tflite` из [Face Landmarker Bundle](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task) через `unzip`.

```sh
python experiments/avatar/render_avatar.py /path/to/slovo-privet.mp4 /tmp/privet.mp4 \
  --models /path/to/weights --start 0.134 --end 1.95
```

Перед подключением сверьте с записью обе кисти, пальцы, траекторию и важную мимику. Покажите MP4 носителю РЖЯ без подписи, имени файла и контекста. Спросите: «Какой жест вы увидели?» Установите `reviewed: true` после узнавания. Условия использования записей опубликованы в [лицензии Slovo](https://github.com/hukenovs/slovo/blob/main/license/en_us.pdf).

Для будущей фразы используйте одну целую запись РЖЯ, чтобы сохранить смысл и грамматику.
