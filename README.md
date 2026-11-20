Motorcycle Mileage Prediction

A Machine Learning project that predicts motorcycle fuel efficiency (KM/L) using motorcycle specifications, riding behaviour, road conditions, maintenance information, and trip details.

Project Overview

The objective of this project is to develop a regression-based Machine Learning model that estimates the mileage of a motorcycle based on multiple influencing factors.

The trained model is integrated into a Flask web application where users can select their motorcycle, provide trip-related information, and receive a predicted mileage.

Features

Motorcycle brand, model, and year selection

Automatic retrieval of motorcycle specifications from the dataset

Trip-based mileage prediction

Data preprocessing and feature engineering

Multiple Machine Learning model comparisons

Hyperparameter tuning

Complete saved prediction pipeline

Flask-based web interface

Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Flask

Joblib

HTML

CSS

JavaScript

Machine Learning Workflow

Data loading and inspection

Data cleaning

Exploratory Data Analysis (EDA)

Feature engineering

Data preprocessing

Model training and comparison

Hyperparameter tuning

Model evaluation

Pipeline creation and saving

Flask application integration

Models Compared

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

Final Model Performance

The tuned Gradient Boosting Regressor achieved the following results on the test dataset:

Metric

Score

MAE

1.4692

MSE

4.2543

RMSE

2.0626

R² Score

0.9652

The final model pipeline is saved at:

artifacts/final_mileage_pipeline.joblib

Project Structure

Add the final verified project directory tree here.

Installation and Setup

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL

Navigate to the project directory:

cd motorcycle-mileage-prediction

Install dependencies:

pip install -r requirements.txt

Run the Flask application:

python app.py

Open the local address displayed in your terminal.

Screenshots

Add screenshots of:

Exploratory Data Analysis

Model evaluation

Flask prediction interface

Example prediction result

Live Demo

Coming soon.

Future Improvements

Deploy the application online.

Improve prediction accuracy using additional real-world data.

Expand motorcycle specification coverage.

Add more detailed prediction insights.

Author

Mohamed Afham

GitHub: https://github.com/AfhamAI