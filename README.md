# Emotion Recognition with Deep Learning

This project leverages deep learning models to perform facial emotion recognition using the FER-2013 dataset. Two models, VGG16 and EfficientNetB2, were trained, fine-tuned, and evaluated to classify facial expressions into one of seven emotions.

## Dataset

The dataset used for training and validation is the FER-2013 dataset, which can be downloaded from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013). The dataset consists of grayscale images of faces, each labeled with one of seven emotions: anger, disgust, fear, happiness, sadness, surprise, and neutral.

## Directory Structure

The directory structure for the project is as follows:

```
.
├── train/                 # Training images
│   ├── anger/
│   ├── disgust/
│   ├── fear/
│   ├── happiness/
│   ├── neutral/
│   ├── sadness/
│   └── surprise/
├── test/                  # Testing images
├── best_model.keras       # Trained model using EfficientNetB2
└── best_modelv2.keras     # Trained model using VGG16
```

## Requirements

The project uses Python 3 and the following libraries:
- numpy
- pandas
- tensorflow
- keras
- matplotlib

## Model Training and Evaluation

### EfficientNetB2 Model

1. **Data Preparation**:
    - The images are loaded and preprocessed using `ImageDataGenerator` with a validation split of 0.2.

2. **Model Architecture**:
    - The base model used is EfficientNetB2, with additional layers for global average pooling, dropout, and dense layers for classification.

3. **Training**:
    - The model is compiled and trained using the Adam optimizer and categorical crossentropy loss function. 
    - Early stopping and learning rate reduction callbacks were utilized to enhance training performance.

### VGG16 Model

1. **Data Preparation**:
    - Similar to the EfficientNetB2 model, images are loaded and preprocessed using `ImageDataGenerator` with a validation split of 0.2, and rescaling is applied.

2. **Model Architecture**:
    - The base model used is VGG16, with additional layers including flattening, dropout, and dense layers for classification.

3. **Training**:
    - The VGG16 model is compiled and trained with the Adam optimizer and categorical crossentropy loss function. 
    - Early stopping and learning rate reduction callbacks were utilized to enhance training performance.

## Model Evaluation and Fine-Tuning

- Both models were trained for an initial 20 epochs and subsequently fine-tuned for an additional 20 epochs to improve accuracy and performance.
- The training history, including accuracy and loss over epochs, was plotted to visualize the model's performance.

## Results

The models were evaluated based on their performance on the validation set. The trained models, `best_model.keras` and `best_modelv2.keras`, represent the final versions of EfficientNetB2 and VGG16 models, respectively.

## Usage

To use the trained models for emotion recognition on new images, load the appropriate model and preprocess the input images to match the training conditions (96x96 pixels, RGB color mode).
