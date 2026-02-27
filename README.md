# Image Classification using CNN (Fashion-MNIST)

## Project Overview
This project builds a **Convolutional Neural Network (CNN)** to classify images from the **Fashion-MNIST dataset** into **10 clothing categories**.  
It demonstrates **end-to-end deep learning skills**, including data preprocessing, data augmentation, model building, training, evaluation, and visualization.

---

## Objective
- Predict the correct clothing category for each image.  
- Maximize **classification accuracy** while preventing overfitting.

---

## Dataset
- **Dataset Name:** Fashion-MNIST  
- **Number of Classes:** 10  
  - T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot  
- **Training Samples:** 60,000  
- **Test Samples:** 10,000  
- **Source:** Built-in TensorFlow/Keras dataset

---

## Tools & Technologies
- **Programming Language:** Python  
- **Libraries:** TensorFlow, Keras, NumPy, Matplotlib, Seaborn, scikit-learn  
- **Techniques Used:**  
  - CNN Architecture: Conv2D, MaxPooling, Flatten, Dense layers  
  - Dropout for regularization  
  - Early stopping to prevent overfitting  
  - Data augmentation (rotation, zoom, shift, flip)  
  - Model evaluation: Accuracy, Confusion Matrix, Classification Report  
  - Visualization of training curves and predictions

---

## Project Steps
1. **Data Loading & Exploration**  
   - Loaded Fashion-MNIST dataset from Keras  
   - Visualized sample images to understand data  

2. **Data Preprocessing**  
   - Normalized pixel values (0–1)  
   - Added channel dimension for CNN input  

3. **Data Augmentation**  
   - Applied rotation, zoom, shifts, and flips to expand dataset  

4. **Model Building**  
   - Constructed a CNN with 2 convolutional layers, max-pooling, fully connected layer, and dropout  

5. **Model Training**  
   - Trained model using augmented data  
   - Applied **early stopping** to prevent overfitting  

6. **Model Evaluation**  
   - Evaluated test accuracy  
   - Generated **confusion matrix** and **classification report**  
   - Visualized **predictions and training curves**  

7. **Model Saving**  
   - Saved trained model for future deployment

---

## Results
- **Test Accuracy:** ~92%  
- Correctly classified all 10 clothing categories  
- Training curves show **smooth convergence** with early stopping  
- Confusion matrix and classification report validate **model performance**

---

## Project Impact
- Demonstrates **end-to-end deep learning workflow**  
- Shows ability to handle **unstructured image data**  
- Illustrates understanding of **CNN architectures, overfitting prevention, and evaluation metrics**  
- Portfolio-ready and can be extended for **real-world image classification tasks**

---

## Future Improvements
- Use **advanced CNN architectures** (e.g., ResNet, VGG)  
- Apply **hyperparameter tuning** for better performance  
- Implement **Grad-CAM visualization** for interpretability  
- Deploy the model as a **web application or API**

---

## How to Run
1. Clone or download the project.  
2. Install required libraries:  
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
