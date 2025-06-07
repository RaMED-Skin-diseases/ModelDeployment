# 🧠 Skin Disease Detection API (FastAPI + TensorFlow)

This is a FastAPI backend deployedd on google cloud for predicting skin diseases using a TensorFlow model with ResNet50 feature extraction. Users can upload an image, and the API will return the most likely skin condition from a predefined set of classes.

---

## 🚀 Features

- ✅ Accepts image uploads via `POST /`
- 🧠 Uses **ResNet50** for feature extraction
- 🔬 Pretrained TensorFlow model (`.h5`) for classification
- ⚡ Fast and lightweight API built with **FastAPI**
- 🧪 Returns class label and prediction probability

---

## 🧰 Supported Skin Disease Classes

The API predicts the following conditions:

- Eczema  
- Melanoma  
- Atopic Dermatitis  
- Basal Cell Carcinoma  
- Melanocytic Nevi  
- Benign Keratosis  
