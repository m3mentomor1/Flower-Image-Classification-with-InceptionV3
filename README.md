# Flower-Image-Classification-with-InceptionV3

### 🧐 I. Overview
This repository contains a deep learning project for flower image classification using the InceptionV3 CNN architecture. The project leverages transfer learning on the InceptionV3 pre-trained model, fine-tuning it on a specific dataset of flower images.

----------------------

### 🗂️ II. Dataset
#### Flower Classification
- This dataset contains images of 14 distinct flower types: carnation, iris, bluebells, golden English, roses, fallen nephews, tulips, marigolds, dandelions, chrysanthemums, black-eyed daisies, water lilies, sunflowers, and daisies. It includes 13,618 training images and 98 validation images, with a total size of 202MB.

**Download Dataset Here:** https://www.kaggle.com/datasets/marquis03/flower-classification/data

----------------------

### 🧑🏻‍💻 III. Learning Approach Used
#### Supervised Learning
- a type of learning in machine learning that involves training an algorithm on labeled data, where input samples are paired with corresponding output labels. The objective is to learn a mapping from input data to correct output labels by adjusting internal parameters during training, minimizing the difference between predicted outputs and true labels.
##
**Why Supervised Learning?**

The model's training adopted a supervised learning approach, as the dataset included explicit labels for each image. 

----------------------

### 🧮 IV. Algorithm Used
#### Neural Networks 
- also known as Artificial Neural Networks (ANNs), are a class of algorithms inspired by the structure and functioning of the human brain. It consists of interconnected nodes organized into layers. These layers typically include an input layer, one or more hidden layers, and an output layer. Each connection between nodes has an associated weight, and nodes within a layer may have activation functions.
##
**Why Neural Networks?**

----------------------

### Architecture Used: 
#### Convolutional Neural Network (CNN)
- 

----------------------

### Base Model: 
#### Inception v3
- A convolutional neural network (CNN) architecture developed by Google as part of the GoogLeNet project, it represents the third edition of Google's Inception Convolutional Neural Network introduced during the ImageNet Recognition Challenge. It has 48 layers & has been pre-trained on the ImageNet dataset, containing millions of labeled images spanning thousands of classes. Additionally, it is well-known for its application in computer vision projects.

Download Weights File Here: https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

----------------------

### Dependencies: 
- **OpenCV** - an open-source computer vision library known for its use in image processing and computer vision projects.
- **Keras** - a high-level neural networks API written in Python, capable of running on top of TensorFlow. This library is utilized for building and training deep learning models.
- **NumPy** - a library for scientific computing with Python.
- **Pandas** - a data analysis library that provides data structures like DataFrames for efficient data handling. 
- **Matplotlib** - a comprehensive library for creating static, interactive, and animated plots in Python, facilitating data visualization tasks.
- **Seaborn**
- **TensorFlow**
- **scikit-learn** 
- **Pillow**
