<div align="center">

# 🏍️ Motorcycle Mileage Prediction

**Predict motorcycle fuel efficiency (KM/L) from specs, riding behaviour, road conditions, maintenance and trip details.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask&logoColor=white)
![R² Score](https://img.shields.io/badge/R%C2%B2-0.9652-brightgreen)

</div>

---

## 📖 Overview

This project builds a regression-based Machine Learning model that estimates a motorcycle's mileage from multiple influencing factors.

The trained model is integrated into a **Flask web app** where users select their motorcycle, enter trip details, and get a predicted mileage instantly.

## ✨ Features

- 🏷️ Motorcycle brand, model and year selection
- ⚙️ Automatic retrieval of motorcycle specs from the dataset
- 🛣️ Trip-based mileage prediction
- 🧹 Data preprocessing and feature engineering
- 📊 Comparison of multiple ML models
- 🎯 Hyperparameter tuning
- 📦 Complete saved prediction pipeline
- 🌐 Flask-based web interface

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python |
| **Data & ML** | Pandas, NumPy, Scikit-learn, Joblib |
| **Visualization** | Matplotlib, Seaborn |
| **Web App** | Flask, HTML, CSS, JavaScript |

## 🔄 Machine Learning Workflow

1. Data loading and inspection
2. Data cleaning
3. Exploratory Data Analysis (EDA)
4. Feature engineering
5. Data preprocessing
6. Model training and comparison
7. Hyperparameter tuning
8. Model evaluation
9. Pipeline creation and saving
10. Flask application integration

## 🤖 Models Compared

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor ✅ *(selected)*

## 📈 Final Model Performance

The tuned **Gradient Boosting Regressor** achieved these results on the test set:

| Metric | Score |
|---|---|
| MAE | 1.4692 |
| MSE | 4.2543 |
| RMSE | 2.0626 |
| **R² Score** | **0.9652** |

The final pipeline is saved at `artifacts/final_mileage_pipeline.joblib`.

## 📁 Project Structure

```text
bike-mileage-predictor/
├── artifacts/
│   └── final_mileage_pipeline.joblib
├── app.py
├── requirements.txt
└── README.md
```

> Replace this with your actual directory tree (run `tree` in the project folder).

## 🚀 Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/AfhamAI/bike-mileage-predictor.git
cd bike-mileage-predictor
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the Flask app**

```bash
python app.py
```

**4. Open the local address shown in your terminal** (usually `http://127.0.0.1:5000`).

## 🖼️ Screenshots

<!-- Replace these with real images, e.g. ![EDA](screenshots/eda.png) -->

| Exploratory Data Analysis | Model Evaluation |
|---|---|
| *coming soon* | *coming soon* |

| Prediction Interface | Example Result |
|---|---|
| *coming soon* | *coming soon* |

## 🌍 Live Demo

Coming soon.

## 🔮 Future Improvements

- [ ] Deploy the application online
- [ ] Improve accuracy with additional real-world data
- [ ] Expand motorcycle specification coverage
- [ ] Add more detailed prediction insights

## 👤 Author

**Mohamed Afham**
GitHub: [@AfhamAI](https://github.com/AfhamAI)****
