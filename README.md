# 🧠 WiseBin – AI Trash Classifier

WiseBin is an AI system that classifies waste into **Plastic**, **Paper**, **Aluminium**, or **Other** in real-time using a webcam and a TensorFlow Lite model.

## 🚀 Features
- Runs locally on CPU (no cloud needed)
- Real-time webcam classification
- Compact MobileNetV2 model
- Exportable to `.tflite` for EdgeTPU or embedded use

## 🧰 Requirements
```
pip install -r requirements.txt
```

## ▶️ Run the AI
```
python main.py
```

Press `q` to exit.

## 📦 Model
You can retrain your own using `train_model_colab.ipynb`, then convert to TFLite with `convert_to_tflite.py`.

## 💡 Future ideas
- Integrate Google Coral Edge TPU  
- Add hardware sorter (servo / conveyor)  
- Deploy on Raspberry Pi 5
