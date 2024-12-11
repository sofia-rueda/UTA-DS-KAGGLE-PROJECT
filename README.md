# UTA-DS-KAGGLE-PROJECT
---

## Mushroom Classification

### One Sentence Summary

This repository holds an attempt to classify mushrooms into edible or poisonous categories using a decision tree on the [Mushroom Classification Dataset](https://www.kaggle.com/datasets/uciml/mushroom-classification) from Kaggle.

---

## Overview

### Definition of the Task 

The task is to classify mushrooms into two categories: poisonous and edible, based on their features. The dataset contains descriptions of 23 species of gilled mushrooms, with features such as cap shape, color, and odor. The challenge is to identify which features are the most indicative of a poisonous mushroom and to determine which machine learning model performs best for this classification task.

---

### My Approach

The approach in this repository formulates the problem as a classification task, using a decision tree as the model. Decision trees are particularly suitable for this dataset because the features are categorical in nature, and the model can easily handle these without requiring significant preprocessing.

---

### Summary of the Performance Achieved

1. **Accuracy Scores**:
   - The accuracy for each fold of the 5-fold cross-validation is:
     - Fold 1: 88.18%
     - Fold 2: 100%
     - Fold 3: 100%
     - Fold 4: 100%
     - Fold 5: 92.61%

   These scores suggest that the model performs well across most folds, but Fold 1 has a notably lower accuracy, indicating potential variability in the data distribution or performance.

2. **Mean Accuracy**:
   - The average accuracy across all folds is **96%**, demonstrating that the model is generally effective at classifying the data.

3. **Standard Deviation**:
   - The standard deviation of **0.05** (5%) indicates some variability in the model’s performance across folds. While this isn't excessively high, it suggests that the model may struggle with certain subsets of the data.

---
## Summary of Work Done

### **Data**

1. **Type**
   - **Input**: The dataset consists of a CSV file containing features such as cap shape, color, odor, gill size, etc., for different mushroom species.
   - **Output**: The target variable is the classification label: "edible" or "poisonous" (with the "unknown" class combined with "poisonous").

2. **Size**
   - The dataset contains 8124 instances, each with 22 features.

3. **Instances (Train, Test, Validation Split)**
   - The dataset is divided into training, validation, and test sets. Typically, 80% of the data is used for training, and the remaining portion is split for testing and validation.
   - Cross-validation is performed using 5 folds.

---
### **Data Loading and Initial Look**

1. **Load the Data**
   - The dataset is loaded from the CSV file.

2. **Count the Number of Rows (Data Points) and Features**
   - The dataset consists of 8124 rows and 22 features.

3. **Check for Missing Values**
   - The dataset is checked for any missing values.

4. **Feature Table**  
   - A table is created for each feature, specifying whether it is categorical or numerical, its value range or categories, the number of missing values, and whether there are any outliers.

5. **Outliers**
   - Outliers are generally not a concern for categorical data but can be assessed for unusual feature distributions.

6. **Class Imbalance**
   - The distribution of the target variable ("edible" vs "poisonous") is checked to identify if the dataset has a significant class imbalance.

---
### **Data Cleaning and Preprocessing**

1. **Data Cleaning**
   - Missing or erroneous data is handled if necessary. Irrelevant columns (such as IDs or metadata) are dropped.

2. **Feature Encoding**
   - Categorical features are transformed into numerical representations using one-hot encoding and label encoding.

3. **Rescaling**
   - For decision tree models, rescaling is generally not necessary, as decision trees can handle categorical and numerical data without scaling.

---
### **Data Visualization**

- **For Classification**:  
  Bar plots and heatmaps are used to visualize the percentage distribution of each feature across the two classes (edible vs. poisonous). These visualizations help identify patterns in feature distributions between the classes.

- **Class-Based Feature Distribution**:
   - Bar plots and heatmaps are used to compare how the features (such as odor, cap shape, etc.) are distributed across the classes. These visualizations allow for the identification of which features are more strongly associated with each class.

---
### **Machine Learning**

#### **Problem Formulation**
   - Unnecessary columns are removed, and the target variable is correctly encoded for classification. I choose to not removed any columns.

#### **Train ML Algorithm**
   - The dataset is split into training and test sets, and a decision tree classifier is trained on the data. Cross-validation is performed to evaluate the model's performance.

---
## **How To Reproduce Results**
#### **1. Data Cleaning**
   - **Missing Values**:  
     First, check for any missing values in the dataset. If missing values are found, handle them by either:
     - Dropping rows or columns with missing data.
     - Imputing missing values (e.g., using the mode for categorical features and median/mean for numerical features).
   
     Since there are no significant missing values in my dataset, this step is not necessary.

   - **Duplicate Rows**:  
     There was no duplicate rows found.

   - **Feature Encoding**:  
     - **Label Encoding**: Since my dataset has categorical features, these should be encoded for the model to understand them. Label encoding is used to convert each category into a unique integer. For instance, "edible" could be encoded as 0, and "poisonous" as 1.

#### **2. Feature Scaling**
   - **Rescaling**:  
     Decision trees do not require feature scaling, as they split the data based on feature values, not the scale. Therefore, I do not need to rescale my features for this model.

   - **One-Hot Encoding for Categorical Features**:  
     If you are using any algorithms that require one-hot encoding (e.g., logistic regression or neural networks), you could apply one-hot encoding. However, with decision trees, label encoding is sufficient.
#### **3. New CSV File**
   - Download a new CSV file after making the alterations of encoding so the work can be saved and transferred to a new .ipynb file.

#### **4. Data Split (Train, Validation, Test)**
   - Split the dataset into three parts:
     - **Training Set**: Typically 80% of the data.
     - **Validation Set**: For tuning hyperparameters and evaluating the model during training (usually 10-15%).
     - **Test Set**: For the final evaluation (10-15%).

### **Machine Learning**

#### **1. Problem Formulation**
   - **Target Variable**:  
     The target variable in this classification problem is the **class** of the mushroom (edible or poisonous), which has been encoded with label encoding. The values are:
     - **0**: Edible
     - **1**: Poisonous

   - **Feature Selection**:  
     I need to select the relevant features for the decision tree model. Features that may include:
     - Cap shape
     - Cap surface
     - Gill size
     - Odor, etc.
     Remove any irrelevant columns, such as ID numbers or already encoded features (e.g., from one-hot encoding), to avoid redundancy.

   - **Data Splitting**:  
     After cleaning and encoding the data, split it into training, validation, and test sets. I used  `train_test_split` from `sklearn.model_selection` to divide the data.

#### **2. Train ML Algorithm**
   - **Decision Tree Classifier**:  
     A decision tree is a suitable algorithm for classification tasks like this, especially with categorical features. I used the `DecisionTreeClassifier` from `sklearn.tree` to train the model.
     
#### **3. Training the Model**
   - Train the model on the training dataset.  
   - I used 5-fold cross-validation to evaluate the model's performance on different subsets of the data and avoid overfitting.

#### **4. Evaluate Model Performance**
   - After training, evaluate the model using the validation set.  
   - The **accuracy score** is a typical metric for classification problems, but I also used other metrics like precision, recall, and F1-score to evaluate performance more thoroughly.

   - Use the validation set to assess how well the model generalizes to unseen data.


#### **5. Performance Metrics**
   - For classification tasks, the common performance metrics include:
     - **Accuracy**: The percentage of correct predictions.
     - **Precision**: The percentage of true positives among all positive predictions.
     - **Recall**: The percentage of true positives among all actual positive instances.
     - **F1-Score**: A balanced metric between precision and recall.

#### **6. Visualizing Model Performance**
   - **Feature Importance**:  
     Since I am using a decision tree, the model will provide **feature importance** scores. Visualize these to see which features contribute the most to the classification.

---
## **Overview of files in repository**
The project follows a standard structure where different tasks such as data preprocessing, model training, visualization, and performance evaluation are organized into separate scripts or notebooks.

* data loading & initial look.ipynb : Load the data, undergo data understanding, and loading a feature table.
* data cleaning & preprocessing.ipynb: Functions for data cleaning, and preprocesses the input dataset and prepares the data for machine learning.
* data visualization.ipynb: Creates various visualizations to understand the dataset.
* machine learning.ipynb: Defines and contains the machine learning model (Decision Tree), trains the Decision Tree Classifier, and evaluates the performance of the trained model.

Certainly! Here's the **Software Setup** section without the code:

---

### **Software Setup for Mushroom Classification Project**

In this section, we will outline the necessary packages and how to set up your environment to run the Mushroom Classification project.

---

### **Required Packages**

To run this project, you will need the following packages:

1. **pandas**: This package is essential for data manipulation and analysis. It provides tools for reading data, cleaning, and preparing it for machine learning tasks.
2. **numpy**: A package for numerical operations, used to handle arrays and perform mathematical calculations.
3. **matplotlib**: A plotting library to create static, animated, and interactive visualizations such as bar plots, line charts, and histograms.
4. **seaborn**: Built on top of matplotlib, this library is used for advanced visualizations, including heatmaps and more complex statistical visualizations.
5. **scikit-learn**: A powerful library for machine learning, providing various algorithms (like Decision Tree) for classification, regression, and clustering tasks.
6. **LabelEncoder**: Part of the scikit-learn package, this is used for encoding categorical variables into numerical values that can be used by machine learning models.

---

### **Installation Instructions**

Follow the steps below to set up your environment:

1. **Create a Virtual Environment (Optional but Recommended)**:
   - It’s recommended to use a virtual environment to manage dependencies for your project. This helps prevent conflicts with other projects.
   - If you're using a virtual environment, activate it before proceeding with the installation of packages.

2. **Install Required Packages**:
   - Use a package manager like `pip` to install the required libraries.
   - If you're using a standard Python installation, simply run `pip install` for each of the required packages.

3. **Verify Installation**:
   - After installation, you can verify that the packages have been installed correctly by checking their versions. This can be done by importing each package and checking the version numbers.

4. **Using Jupyter Notebooks (Optional)**:
   - If you're using Jupyter Notebooks, ensure you have it installed as well. You can do so by installing the `notebook` package. Once installed, you can start Jupyter Notebooks from the command line.

---

### **How to Install and Use the Package**

1. **Clone the Repository (if applicable)**:
   - If you have the Mushroom Classification project hosted on GitHub or another platform, you can clone the repository to your local machine.

2. **Install Dependencies**:
   - If the project contains a `requirements.txt` file (a list of all the necessary packages), you can install all dependencies at once by using the `pip install -r requirements.txt` command.

3. **Run the Project**:
   - Once everything is installed, you can open the provided Jupyter Notebook files (or Python scripts) and start working through the project.

---

### **Optional: Using Google Colab**

If you don’t want to manage the installations on your local machine, you can also use **Google Colab**, which provides free access to a virtual environment with most of these libraries already installed. You can simply upload the dataset and start running the code directly in Colab.

---

This setup will ensure that you have all the tools required to run the Mushroom Classification project and begin exploring the dataset and models.

