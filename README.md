# diabetes_prediction
# 🩺 Diabetes Risk Prediction Using Lifestyle and Clinical Data

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--Learn-green)
![License](https://img.shields.io/badge/License-MIT-purple)

A machine learning-powered web application that predicts the risk of diabetes using lifestyle habits and clinical health indicators. Built with **Random Forest**, trained on real-world health survey data, and deployed using **Streamlit** for an interactive user experience.

---

## 📌 Project Overview

This project aims to **identify individuals at high risk of diabetes** using demographic, behavioral, and clinical features such as BMI, physical activity, smoking, and general health. Early risk detection enables timely lifestyle interventions and medical follow-up.

- **Model**: Random Forest Classifier (handles non-linear relationships and feature interactions)
- **Dataset**: Behavioral Risk Factor Surveillance System (BRFSS) 2015 — cleaned and preprocessed
- **Deployment**: Streamlit for an intuitive web interface
- **Goal**: Predict diabetes risk from 18 health indicators with high accuracy and interpretability

---

## 🧪 Dataset

- **Source**: [Kaggle - Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- **Original Survey**: CDC’s Behavioral Risk Factor Surveillance System (BRFSS) 2015
- **Samples**: ~250,000+ adult respondents
- **Features**: 21 health-related variables
- **Target**: `Diabetes` (binary: 0 = No, 1 = Yes)

### 🔢 Features Included
| Feature | Description |
|--------|-------------|
| `Diabetes` | Target variable (1 = Yes, 0 = No) |
| `HighBP` | High Blood Pressure |
| `HighChol` | High Cholesterol |
| `CholCheck` | Cholesterol check in past 5 years |
| `BMI` | Body Mass Index |
| `Smoker` | Smoked at least 100 cigarettes |
| `Stroke` | History of stroke |
| `HeartDiseaseorAttack` | Coronary heart disease or heart attack |
| `PhysActivity` | Physical activity in past 30 days |
| `Fruits`, `Veggies` | Daily fruit/vegetable consumption |
| `HvyAlcoholConsump` | Heavy alcohol consumption (men >14, women >7 drinks/week) |
| `AnyHealthcare` | Any healthcare coverage |
| `NoDocbcCost` | Could not see doctor due to cost |
| `GenHlth` | General health (1=Excellent → 5=Poor) |
| `MentHlth`, `PhysHlth` | Days of poor mental/physical health (past 30 days) |
| `DiffWalk` | Difficulty walking |
| `Sex` | Gender (0=Female, 1=Male) |
| `Age` | Age category (1=18-24 → 13=80+) |
| `Education`, `Income` | Socioeconomic status |

> 🔍 **Note**: The original dataset includes `2` for "pre-diabetes". This project maps `2 → 0` to focus on **diagnosed diabetes**.

---

## 🛠️ Model Training

- **Algorithm**: Random Forest with `class_weight='balanced'`
- **Train/Test Split**: 80%/20% with stratification
- **Evaluation Metrics**:
  - Accuracy: ~83.8%
  - ROC-AUC: ~0.86
  - F1-Score (Diabetic class): ~0.60
- **Key Features**: `GenHlth`, `BMI`, `Age`, `PhysHlth`, `DiffWalk`

> ✅ The model handles class imbalance and provides robust probability estimates for risk scoring.

---

## 🌐 Streamlit Web App

A user-friendly interface allows individuals to input their health data and receive an instant diabetes risk prediction.

### 💡 Features
- Interactive form with real-time risk estimation
- Probability score and visual progress bar
- Clear interpretation: "High Risk" or "Lower Risk"
- Model confidence via predicted probability

### 🖼️ App Screenshot 
>   
> (<img width="386" height="683" alt="Screenshot 2025-08-11 000124" src="https://github.com/user-attachments/assets/f29503b2-65a5-4bfa-bfea-d580b51e1c38" />
)

---

## 📦 Project Structure
```
diabetes-prediction/
│
├── data/
│ └── diabetes_binary_health_indicators_BRFSS2015.csv
├── models/
│ └── trained_model.pkl
├── app.py # Main Streamlit app
├── train_model.py # Model training & evaluation script
├── requirements.txt # Python dependencies
└── README.md # This documentation file
```

---

---

## ▶️ How to Run Locally

### Clone the Repository
```bash
git clone https://github.com/sudheersivakumar/diabetes_prediction
cd diabetes_prediction
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
### Requirements
```bash
streamlit
pandas
numpy
scikit-learn
joblib
```
### Requirements-Installation
```bash
pip install streamlit pandas numpy scikit-learn joblib
```
## Future Enhancements
- Add ROC curve and confusion matrix in the app
- Show top contributing risk factors for each user
- Multiclass mode: Predict No Diabetes / Pre-Diabetes / Diabetes
- Generate PDF risk report
- Add lifestyle tips based on input (e.g., "Increase physical
activity")
- Support for multiple languages
