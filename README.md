# project---3-
Forest Cover Type Prediction Using Machine Learning

🌲 Forest Cover Type Prediction
📌 Project Overview

This project aims to predict the forest cover type (type of trees/vegetation) based on geographical and environmental features using Machine Learning.

🎯 Objective

To build a model that can accurately classify forest areas into different cover types using features like elevation, soil type, and wilderness area.

📊 Dataset
The dataset contains cartographic variables such as:
Elevation
Aspect
Slope
Distance to water, road, fire points
Soil type (categorical)
Wilderness area (categorical)
Target variable: Cover_Type (multi-class classification)


🔍 Steps Performed


1️⃣ Data Preprocessing
Checked missing values (none found)
Converted categorical features using encoding
Scaled/normalized features where needed


2️⃣ Exploratory Data Analysis (EDA)
Used:
Distribution plots → to understand feature spread
Correlation heatmap → to find relationships
Boxplots → to detect outliers
Identified important features affecting prediction


3️⃣ Feature Engineering
Removed less useful features
Selected important variables using model-based importance


4️⃣ Model Building
Trained classification models like:
Random Forest
(Optional: XGBoost / Decision Tree)
Chosen model: Random Forest Classifier
(because it gives better accuracy and handles overfitting well)


5️⃣ Hyperparameter Tuning

Used techniques like:

RandomizedSearchCV

Example parameters:

n_estimators
max_depth
min_samples_split
min_samples_leaf

This helped improve model performance.



6️⃣ Model Evaluation
Evaluated using:
Accuracy score
Confusion matrix
Classification report


7️⃣ Model Saving
Saved trained model using pickle
Used later in deployment (e.g., Streamlit app)


🚀 Deployment
Built a simple UI using Streamlit
User inputs environmental features
Model predicts the forest cover type


🛠️ Tools & Technologies
Python
Pandas, NumPy
Scikit-learn
Matplotlib, Seaborn
Streamlit
Pickle


📌 Conclusion

The model successfully predicts forest cover types based on input features, helping in environmental analysis and decision-making.




