## Dataset
The dataset used in this project can be accessed on Kaggle: [Caltech Dataset](https://www.kaggle.com/datasets/imbikramsaha/caltech-101)

# Introduction 
This project demonstrates the fine-tuning of a Vision Transformer (ViT) model for image classification using the Caltech 101 dataset. The Caltech 101 dataset contains around 9,000 images across 102 categories, including 101 object categories like animals, flowers, and vehicles, along with one background category. It is a popular dataset for image classification tasks due to its diverse and well-annotated images. In this project, we fine-tune a pre-trained ViT model to classify images into their respective categories. Vision Transformers leverage self-attention mechanisms to process image patches, making them highly effective for image recognition tasks. The goal of this project is to achieve accurate and efficient classification of the images in the Caltech 101 dataset using state-of-the-art deep learning techniques.

## step 1 (Step up the Environment)
This command installs essential libraries for the project: **Transformers**, for working with pre-trained models like Vision Transformers (ViT); **Torch**, the core library for deep learning with PyTorch; and **Torchvision**, which provides tools for image processing and pre-trained computer vision models. These libraries are crucial for tasks like model fine-tuning and image classification in this project.

## step 2 (unzip the file)
 This code extracts the contents of a ZIP file. It specifies the file path (`zip_path`) and the folder to extract into (`extract_path`). Using Python's `zipfile` module, it opens the ZIP file and extracts all its contents to the specified directory, ensuring easy access to the data.

## step 3 ( import libraries)
Key libraries were imported to utilize pre-trained models (like ViT), preprocess the dataset, and 
manage training.
• ViTForImageClassification: For using the ViT model.
• ViTFeatureExtractor: To preprocess images into a format suitable for ViT.
• Trainer: A convenient training utility for supervised learning tasks

## step 4 (preposses the dataset)
A preprocessing function was applied to the dataset to ensure all images and labels were in a 
format compatible with ViT. Images were resized, normalized, and tokenized for input to the 
transformer. This step also added preprocessed pixel values as a new column in the dataset for 
easier access during training

## step 5 ( Split Dataset into Train/Test Sets)
The dataset was split into training and validation subsets. The training set is used to fine-tune the 
model, while the validation set ensures the model's generalization ability is evaluated during and 
after training

## Step 6: Load Pre-Trained model
The pre-trained ViT model (base version) was loaded with its weights initialized from training on 
ImageNet. The model's output layer was customized to handle 102 output classes, matching the 
Caltech 101 dataset.

## step 7 Set up the training arguments:
The training configuration included:
• Learning Rate: Controlled the step size for weight updates.
• Batch Size: Defined the number of samples processed together.
• Epochs: Number of times the entire training data was passed through the model.
• Evaluation Strategy: Monitored validation performance during training.
These arguments optimize the model's performance and ensure training efficiency.

## step 8 Train the model
The Trainer utility was used to train the model on the preprocessed training set. This process 
updated the model's weights to fit the Caltech 101 dataset. Training logs showed metrics like loss 
and accuracy at each step, allowing progress monitoring

## Step 9: Visualize result 
After training, the model was evaluated on the validation set to measure accuracy. The fine-tuned 
ViT achieved 96% accuracy on the Caltech 101 dataset, showcasing its ability to adapt to a new 
domain while retaining the benefits of pre-training.

## Step 10: Testing
This step involves testing the model's inference capabilities by using a randomly chosen image 
from an external source (Google). The image is preprocessed and fed into the trained Vision 
Transformer model to observe its classification results. This verifies the model's ability to 
generalize to unseen data.
