# WiseBin – AI Trash Classifier

WiseBin is an AI system that classifies waste into **Plastic**, **Paper**, **Aluminium**, or **Other** in real-time using a webcam and a TensorFlow Lite model. It was trained on the [Trashnet Dataset](https://github.com/garythung/trashnet)

## Features
- Runs locally on CPU (no cloud needed)
- Real-time webcam classification
- Compact MobileNetV2 model
- Exportable to `.tflite` for EdgeTPU or embedded use

## Requirements
```
pip install -r requirements.txt
```

## Run the AI
```
python main.py
```

Press `q` to exit.
