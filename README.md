 Image Classification using CNN (Fashion-MNIST)
Project Overview

This project focuses on building a Convolutional Neural Network (CNN) to classify images from the Fashion-MNIST dataset into 10 clothing categories. The model demonstrates key deep learning skills, including preprocessing, data augmentation, model training, evaluation, and visualization.

Objective

Predict the correct clothing category for each image in the dataset, aiming to maximize classification accuracy while preventing overfitting.

Dataset

Dataset Name: Fashion-MNIST

Number of Classes: 10 (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

Training Samples: 60,000

Test Samples: 10,000

Source: Built-in TensorFlow/Keras dataset

Tools & Technologies

Programming Language: Python

Libraries: TensorFlow, Keras, NumPy, Matplotlib, Seaborn, scikit-learn

Techniques Used:

CNN Architecture (Conv2D, MaxPooling, Flatten, Dense layers)

Dropout for regularization

Early stopping to prevent overfitting

Data augmentation (rotation, zoom, shift, flip)

Model evaluation (Accuracy, Confusion Matrix, Classification Report)

Visualization of training curves and predictions

Project Steps

Data Loading & Exploration

Loaded Fashion-MNIST dataset from Keras

Visualized sample images to understand data

Data Preprocessing

Normalized pixel values (0-1)

Added channel dimension for CNN input

Data Augmentation

Applied rotation, zoom, shifts, and flips to enhance training dataset

Model Building

Constructed a CNN with two convolutional layers, max-pooling, fully connected layer, and dropout

Model Training

Trained the model using augmented data

Applied early stopping to prevent overfitting

Model Evaluation

Evaluated accuracy on test set

Generated confusion matrix and classification report

Visualized predictions on test images

Model Saving

Saved the trained model for future deployment

Results

Test Accuracy: ~92%

Successfully classified 10 different clothing categories

Training curves show smooth convergence with early stopping

Confusion matrix and classification report validate model performance

Project Impact

Demonstrates end-to-end deep learning workflow

Shows ability to handle image data (unstructured data)

Illustrates understanding of CNN architectures, overfitting prevention, and evaluation metrics

Portfolio-ready and can be extended for real-world image classification tasks

Future Improvements

Apply more complex CNN architectures (e.g., ResNet, VGG)

Use Hyperparameter tuning for better performance

Implement Grad-CAM visualization to interpret predictions

Deploy the model as a web application or API

How to Run

Clone or download the project

Install required libraries:

pip install tensorflow numpy matplotlib seaborn scikit-learn

Run the Python notebook or script

Explore outputs: accuracy, confusion matrix, sample predictions
