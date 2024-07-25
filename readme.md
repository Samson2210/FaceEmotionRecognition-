# Face Emotion Recognition 

## Overview

This project focuses on real-time face emotion recognition using Convolutional Neural Networks (CNN). The model is trained on the FER2013 dataset from Kaggle and utilizes OpenCV for capturing faces and classifying emotions in real-time.


## Dataset

The project uses the FER2013 dataset, which consists of grayscale images of faces, each labeled with an emotion. The dataset is available for download on Kaggle.

## Dependencies

- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Install the dependencies using:

```bash
pip install tensorflow keras opencv-python numpy pandas matplotlib scikit-learn
```

## Model Architecture

The CNN model consists of several convolutional layers with Batch Normalization, Activation, MaxPooling, and Dropout, followed by fully connected layers and an output layer with softmax activation for emotion classification.


## Conclusion

This project demonstrates a practical implementation of face emotion recognition using deep learning. The trained CNN model can effectively classify emotions in real-time, providing a foundation for further enhancements and applications.