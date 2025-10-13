# Pothole Image Classification – South Africa

**Author:** Rupesh Bhardwaj  
**Project Type:** Image Classification / Deep Learning / Computer Vision  
**Dataset:** Potholes images from South Africa  

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Project Structure](#project-structure)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Model Training](#model-training)  
7. [Evaluation](#evaluation)  
8. [Results](#results)  
9. [Future Work](#future-work)  
10. [License](#license)  

---

## Project Overview
This project focuses on detecting and classifying potholes in road images from South Africa. Using deep learning techniques, specifically convolutional neural networks (CNNs), the model can identify images with potholes accurately, which can aid in road maintenance and safety monitoring.  

**Key Features:**  
- Image preprocessing pipeline  
- Training on labeled pothole dataset  
- Model evaluation using accuracy, precision, and recall  
- Prediction on new/unseen images  

---

## Dataset
The dataset contains images of roads with and without potholes. It is split into:  
- `train.csv` – Image paths and labels for training  
- `test.csv` – Image paths for prediction (unlabeled)  
- `images/` – Folder containing all images  

**📂 Dataset**  
Dataset hosted on Hugging Face:  
➡️ [rupesh002/Patholes_Dataset](https://huggingface.co/datasets/rupesh002/Patholes_Dataset)

**🧠 Trained Model**  
Model weights hosted on Hugging Face:  
➡️ [rupesh002/pothole_detection_model](https://huggingface.co/rupesh002/pothole_detection_model)

---

## Project Structure
MIIA_Pothole_Image_Classification_SouthAfrica/
│
├── images/ # Raw images
├── train.csv # Training data with labels
├── test.csv # Test data for prediction
├── notebooks/ # Jupyter notebooks for EDA, preprocessing, training
├── models/ # Saved trained models
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── main.py # Main script for preprocessing, training, prediction

yaml
Copy code

---

## Installation
Clone the repository and install the required dependencies:  
'''bash
git clone https://github.com/Rupeshbhardwaj002/MIIA_Pothole_Image_Classification_SouthAfrica.git
cd MIIA_Pothole_Image_Classification_SouthAfrica
pip install -r requirements.txt

---

## Usage

Data preprocessing – Resize images, normalize, and split into training/validation sets.

Train the model – Run the training script:

python main.py --train


Make predictions on test images – Run:

python main.py --predict

## Model Training

Architecture: Convolutional Neural Network (CNN)

Loss Function: Cross-Entropy Loss

Optimizer: Adam

Metrics: Accuracy, Precision, Recall

Epochs: Configurable in main.py

## Evaluation

The model is evaluated on a validation set using:

Accuracy: Measures overall correctness

Precision: Measures correctly predicted potholes over all predicted potholes

Recall: Measures correctly predicted potholes over all actual potholes

Results are saved and visualized in the notebooks/ folder.

## Results

High accuracy in detecting potholes in South African road images

Confusion matrix and classification report generated for model analysis

Include sample images of predictions or graphs here if possible.

## Future Work

Expand dataset with more diverse road conditions

Implement real-time pothole detection using video feeds

Experiment with transfer learning using pre-trained CNN models

