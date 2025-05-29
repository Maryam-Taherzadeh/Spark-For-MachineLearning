# Spark-For-MachineLearning

This repository contains hands-on examples of machine learning algorithms implemented using **Apache Spark with Python and MLlib**.

## 📌 Project List

1. **Linear Regression**  

###  Project 1: Predict Crew Size for Cruise Ships (Linear Regression with PySpark MLlib)

**📌 Problem Statement:**
Hyundai Heavy Industries wants to predict how many **crew members** a cruise ship will need based on its features. This helps in planning staff, resources, and costs efficiently.

** Objective:**
Use **Linear Regression** with **PySpark MLlib** to build a machine learning model that can estimate the number of crew required based on ship attributes.

** Dataset Overview:**
The dataset (`cruise_ship_info.csv`) includes features like:

* `Age`: Age of the ship
* `Tonnage`: Weight/size of the ship
* `Passengers`: Number of passengers the ship can carry
* `Length`, `Cabins`, `Passenger Density`
* `Crew` (target variable to predict)

** ML Approach:**

* Use **SparkSession** to read and process the data.
* Perform **exploratory data analysis** with `.show()` and `.describe()`.
* Select relevant numeric features.
* Train a **Linear Regression** model using MLlib.
* Evaluate model performance using regression metrics (e.g., RMSE, R²).

** Skills Practiced:**

* Data loading and schema inspection with PySpark
* Feature engineering and selection
* ML model building with Spark MLlib
* Interpretation of linear regression results
* Spark-based big data processing


2. **Logistic Regression**  
   Binary classification problem with Spark.

3. **Decision Tree & Random Forest**  
   Comparison and evaluation using tree-based classifiers in Spark.

4. **K-means Clustering**  
   Unsupervised learning to group data points.

5. **NIP with MLlib**  
   Neural-inspired processing using Spark.

## 🚀 Tools & Libraries
- Apache Spark
- PySpark MLlib
- Python
- Jupyter Notebook


                                                                                                       
