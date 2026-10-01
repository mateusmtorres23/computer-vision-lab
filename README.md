# Computer Vision Lab

A collection of experiments and small projects developed while studying Computer Vision and Machine Learning with Python.

The repository explores different computer vision tasks and model families, ranging from convolutional neural networks to pretrained vision models, multimodal models, image segmentation, object detection, and real-time hand gesture recognition.

## Projects

### LeNet-5 — MNIST

Implementation and experimentation with the LeNet-5 convolutional neural network using PyTorch and the MNIST handwritten-digit dataset.

The experiment includes:

- LeNet-5 architecture implementation
- MNIST dataset loading
- CNN training
- Image classification
- Visualization of convolutional filters

### Image Classification — MobileNetV3

Experiment using a pretrained MobileNetV3 model for image classification.

This project explores transfer learning and inference using models available through the PyTorch ecosystem.

### Image Classification — Gemini

Experiment with multimodal image understanding using Google Gemini.

Images are provided to the model so that their visual content can be interpreted and classified using a multimodal language model.

### Image Segmentation — CLIPSeg

Experiment with text-guided image segmentation using CLIPSeg.

The model receives both an image and a textual description and identifies the regions associated with the requested concept.

### Object Detection — YOLOS

Object-detection experiment using YOLOS, a Transformer-based architecture for detecting and localizing objects in images.

### Hand Gesture Recognition — MediaPipe

A custom hand gesture recognition experiment built with MediaPipe, OpenCV, and scikit-learn.

The project includes:

- Webcam capture with OpenCV
- Hand landmark extraction with MediaPipe
- Custom dataset collection
- Spatial landmark preprocessing
- Gesture label encoding
- Random Forest training
- Model persistence with Joblib

The generated models are used to recognize custom hand gestures.

### Hand Gesture Recognition App

A web application derived from the gesture-recognition experiment.

The application uses FastHTML and WebSockets to process webcam frames, detect hand landmarks, classify gestures with the trained model, and return processed frames and predictions to the interface in real time.

The application also displays gesture information and visual representations of recognized gestures.

## Technologies

- Python
- PyTorch
- torchvision
- Hugging Face Transformers
- MediaPipe
- OpenCV
- scikit-learn
- Google Gemini
- FastHTML
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

## Repository Structure

```text
computer-vision-lab/
├── gesture-recognition-app/
├── gesture-recognition-mediapipe/
├── image-classification-gemini/
├── image-classification-mobilenetv3/
├── image-segmentation-clipseg-rd64/
├── lenet-mnist/
└── object-detection-yolos/
```

Each directory contains an independent experiment focused on a specific computer vision technique or model.

## Future Improvements

- Standardize the feature-processing pipeline between gesture model training and inference.
- Reduce unnecessary datasets and large generated artifacts stored directly in Git.
- Centralize shared image assets used by multiple experiments.
- Add individual documentation for the main experiments.
