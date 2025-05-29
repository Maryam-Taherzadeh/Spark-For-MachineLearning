# Spark-For-MachineLearning

This repository contains hands-on examples of machine learning algorithms implemented using **Apache Spark with Python and MLlib**.

## 📌 Project List

###  Project 1: Predict Crew Size for Cruise Ships (Linear Regression with PySpark MLlib)

📌 Problem Statement:
Hyundai Heavy Industries wants to predict how many **crew members** a cruise ship will need based on its features. This helps in planning staff, resources, and costs efficiently.

   Objective:
Use **Linear Regression** with **PySpark MLlib** to build a machine learning model that can estimate the number of crew required based on ship attributes.

   Dataset Overview:
The dataset (`cruise_ship_info.csv`) includes features like:

* `Age`: Age of the ship
* `Tonnage`: Weight/size of the ship
* `Passengers`: Number of passengers the ship can carry
* `Length`, `Cabins`, `Passenger Density`
* `Crew` (target variable to predict)

 ML Approach:

* Use **SparkSession** to read and process the data.
* Perform **exploratory data analysis** with `.show()` and `.describe()`.
* Select relevant numeric features.
* Train a **Linear Regression** model using MLlib.
* Evaluate model performance using regression metrics (e.g., RMSE, R²).

 Skills Practiced:

* Data loading and schema inspection with PySpark
* Feature engineering and selection
* ML model building with Spark MLlib
* Interpretation of linear regression results
* Spark-based big data processing


 🚢 Project 2: Predicting Titanic Survivors (Logistic Regression with PySpark MLlib)

📌 Problem Statement:
Using the famous Titanic dataset, build a classification model to predict the **probability of passenger survival** based on features like age, sex, ticket class, and fare. This is a **binary classification** task where the goal is to predict `Survived` (0 = No, 1 = Yes).

Objective:
Apply **Logistic Regression** using PySpark's MLlib library to classify passengers as survived or not.

Dataset Overview:
Key columns used for prediction:

* `Pclass`: Ticket class
* `Sex`: Gender (categorical)
* `Age`: Age of passenger
* `SibSp`: Number of siblings/spouses aboard
* `Parch`: Number of parents/children aboard
* `Fare`: Fare paid
* `Embarked`: Port of Embarkation (categorical)

 Steps:

1. **Load CSV data** into Spark DataFrame.
2. **Select relevant features** and handle missing values using `.na.drop()`.
3. **Transform categorical columns** (`Sex`, `Embarked`) using:

   * `StringIndexer`
   * `OneHotEncoder`
4. **Assemble features** into a single vector using `VectorAssembler`.
5. **Split the data** into training and testing sets.
6. **Train the Logistic Regression model** using a Spark `Pipeline`.
7. **Evaluate** the model using `BinaryClassificationEvaluator`.

 Performance:

* Model AUC (Area Under ROC Curve):
* Output includes survival predictions vs actual values for inspection.

 Skills Practiced:**

* Data cleaning and preprocessing with Spark
* Encoding categorical features
* Building machine learning pipelines in PySpark
* Evaluating binary classification models
 


### 🐶 Project 3: Detecting Spoilage in Dog Food Using Tree-Based Models (PySpark)

**📌 Problem Statement:**
A dog food company is investigating early spoilage in its product. The food contains four preservatives (A, B, C, D), and the company suspects one of them may be responsible. Your task is to use **machine learning (Decision Trees or Random Forest)** to identify which preservative has the strongest effect on spoilage.

**Goal:**

* Build a classification model using Spark MLlib tree-based methods.
* Determine **which chemical (A, B, C, or D)** contributes most to spoilage by interpreting feature importance.

**Dataset Overview:**
Each row represents a dog food batch with the following features:

* `A`, `B`, `C`, `D`: Percentage of each preservative
* `Spoiled`: Label (1 = spoiled, 0 = not spoiled)

**Steps:**

1. **Load and explore the dataset** using Spark DataFrame.
2. **Assemble features** into a single vector using `VectorAssembler`.
3. **Train a Decision Tree Classifier** using `DecisionTreeClassifier`.
4. **Evaluate feature importance** from the trained model to find the most predictive preservative.

**Result:**

* Feature importance revealed **Preservative C (feature index 2)** as the **most critical factor** linked to food spoilage.
* Feature importances: `{B: 0.0019, C: 0.9832, D: 0.0149}`

**Skills Practiced:**

* Working with structured data in PySpark
* Feature engineering with Spark’s `VectorAssembler`
* Applying Decision Tree Classifier
* Interpreting feature importances for explainable ML




5. **K-means Clustering**  
   Unsupervised learning to group data points.

6. **NIP with MLlib**  
   Neural-inspired processing using Spark.

## 🚀 Tools & Libraries
- Apache Spark
- PySpark MLlib
- Python
- Jupyter Notebook


                                                                                                       
