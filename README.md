# 🚀 Real-Time Face Recognition with TensorFlow & Keras 🤖

## 🎯 Overview

Welcome to the **Face Recognition** project, where we bring the power of **TensorFlow**, **Keras**, and **OpenCV** to life for **real-time face detection and recognition**! This system allows you to identify faces through a webcam feed, leveraging deep learning and computer vision to predict individuals' identities with impressive accuracy.

This project has been split into three distinct files to streamline your experience. Each file serves a specific purpose, and you can use them individually based on your needs:

1. **train_model.py**: For training your custom face recognition model.
2. **predict_face.py**: For using the trained model to detect and recognize faces in real-time.
3. **plot_result.py**: For visualizing the training and validation performance after the model has been trained.

---

## ✨ Key Features

- **🔍 Real-Time Face Detection**: Detect faces in a webcam feed in real time and classify them instantly.
- **🤖 Powered by TensorFlow & Keras**: Train a **Convolutional Neural Network (CNN)** to recognize multiple individuals.
- **📸 Webcam Integration**: The system uses your webcam to detect and recognize faces live.
- **🔄 Data Augmentation**: Boost the model’s performance and prevent overfitting with data augmentation techniques.
- **💾 Model Saving & Loading**: Train once, save your model, and reuse it for prediction anytime without retraining.
- **📊 Track Training Progress**: Visualize accuracy and loss during training with **matplotlib**.
- **🎮 Fun with Face Recognition**: If the model is unsure about a face, it labels it “Alien” for fun! 👽

---

## 💻 Requirements

Ensure you have Python 3.x and install the following dependencies:

```bash
pip install tensorflow opencv-python matplotlib numpy
```

---

## 🧑‍🤝‍🧑 Dataset

The dataset should be organized in folders, with each folder representing a different person. Each image inside the folder should belong to the corresponding individual. Here's an example structure:

```
dataset/
    person_1/
      image1.jpg
        image2.jpg
        ...
    person_2/
        image1.jpg
        image2.jpg
        ...
    ...
```

---

## 🏗️ How It Works

### 1. **Training the Model** (`train_model.py`)

The **`train_model.py`** file is where you build and train your custom face recognition model. The code in this file will:

- **Load your dataset** and preprocess the images (resize, normalize, etc.).
- **Augment your data** with techniques like rotation, shifting, and flipping to prevent overfitting.
- **Train a CNN model** with multiple convolutional and pooling layers to extract features and classify faces.
- **Save the trained model** as an `.h5` file so you can use it for future predictions.

To run this, simply execute:

```bash
python train_model.py
```

This will train the model and save it to your disk as `face_recognition_model.h5`.

### 2. **Face Prediction in Real-Time** (`predict_face.py`)

After training your model, use the **`predict_face.py`** file for **real-time face detection and recognition** via your webcam. This file:

- Loads the pre-trained model (`face_recognition_model.h5`).
- Uses **OpenCV** to capture video frames from your webcam.
- **Detects faces** in the video feed using OpenCV's Haar Cascade Classifier.
- **Classifies detected faces** using the trained model, displaying the person’s name and confidence score on screen.
- If the model is unsure (confidence below 80%), it will label the face as “Alien” for fun! 👽

Run this file to start recognizing faces:

```bash
python predict_face.py
```

### 3. **Plotting Training Results** (`plot_result.py`)

The **`plot_result.py`** file helps you visualize the training progress. After training, this script will:

- Plot the **training accuracy** and **validation accuracy** over the epochs.
- Plot the **training loss** and **validation loss** to see how well the model is learning and generalizing.

To visualize the results:

```bash
python plot_result.py
```

---

## 🎥 Real-Time Demo

Once the model is trained, execute **`predict_face.py`** to start the real-time face recognition system. The webcam feed will display:

- **Face Detection**: Faces are detected and highlighted with rectangles.
- **Face Recognition**: The model will display the name of the detected person with a confidence score.
- **“Alien” Detection**: If the model isn't confident, it will mark the face as "Alien".

---

## 🔥 Example Output

Here's what you might see on your webcam feed:

```
John Doe (Confidence: 92.34%)
```

If the model isn’t sure:

```
Alien (Confidence: 72.45%)
```

---

## 📊 Visualize Training Results

After training, use **`plot_result.py`** to visualize how well your model performed. The graphs will display:

- **Training & Validation Accuracy**: How well the model is performing on training data and unseen validation data.
- **Training & Validation Loss**: The loss values showing how much the model is improving during training.

---

## 🛠️ Troubleshooting

- **Low Accuracy**: If your model isn't performing well, try increasing the number of epochs, adding more data, or adjusting the training settings.
- **Face Detection Issues**: Poor lighting or unclear images can affect detection. Ensure the faces in the dataset are well-lit and clearly visible.
- **Webcam Access**: If OpenCV can’t access your webcam, check your system’s camera permissions or ensure the correct camera is selected.

---

## 🚀 Next Steps & Enhancements

Here are some exciting ways you can take this project further:

- **🦸‍♂️ Emotion Detection**: Add emotion recognition alongside face identification for a more dynamic system.
- **📦 Use Pre-trained Models**: Integrate models like **VGG16** or **ResNet** for enhanced feature extraction and better performance.
- **⚡ Transfer Learning**: Fine-tune the model for quicker and more efficient training, especially with smaller datasets.

---

## 🎉 License

This project is licensed under the **MIT License**. Feel free to contribute, improve, and build upon it!

---

With this setup, you can easily dive into building your face recognition system, train it, see the results, and have fun with real-time predictions. Ready to go? **Let’s build some AI magic!** ✨🚀

---
