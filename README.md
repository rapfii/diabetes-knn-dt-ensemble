# 🩺 Diabetes Prediction Analysis

> A professional Machine Learning project that predicts diabetes risk using Pima Indians patient medical data. Built with Python, Scikit-Learn, and Ensemble Learning techniques.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5+-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?style=flat-square&logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=flat-square)

---

## 🖼️ Visualizations

| Correlation Heatmap | Confusion Matrix |
|---|---|
| ![heatmap](Output%20Image/Heatmap%20Korelasi.png) | ![matrix](Output%20Image/Confusion%20Matrix%20DS.png) |

---

## ⚡ Key Results

- 🎯 **Best Accuracy:** 79.22%
- 🌲 **Best Model:** Decision Tree Classifier (`max_depth=5`)
- 🔥 **Top Features:** Glucose, BMI, Age
- ⚖️ **Dataset Size:** 768 patient samples
- 🗳️ **Ensemble Method:** Majority Voting (5 models)

---

## 📌 Project Overview

This project uses the **Pima Indians Diabetes Dataset** to predict whether a patient is at risk of **diabetes** or is **healthy** based on medical data (glucose, blood pressure, BMI, insulin, etc.).

The main goal of this project is to demonstrate a clean, end-to-end Machine Learning workflow, from raw data exploration to predictions using Ensemble techniques (Majority Voting).

---

## 🌍 Real-World Impact

This model demonstrates how Machine Learning can support:
- ⚕️ Early detection of diabetes risk in patients
- 💰 Reduction of repeated laboratory testing costs
- 📊 Data-driven decision making in the healthcare industry
- 🏥 More efficient mass screening in healthcare facilities

---

## 📈 Pipeline Execution & Results

Running the full pipeline produces a comprehensive output covering data processing, model training, evaluation, and direct predictions. Below is the detailed breakdown of the project outcomes:

### 1. Library Import & Dataset Loading
- **Main Libraries:** `pandas`, `matplotlib`, `seaborn`, `scikit-learn` (KNeighborsClassifier, DecisionTreeClassifier, VotingClassifier).
- **Dataset:** Loads `diabetes.csv` containing **768 rows** (patients) and 9 columns (8 medical features + 1 target `Outcome`).

### 2. Exploratory Data Analysis (EDA)
- 🌡️ **Correlation Heatmap:** Maps linear relationships between all medical features. The `Glucose` feature has the highest correlation (0.47) to the `Outcome` (diabetes), followed by `BMI` (0.29), `Age` (0.24), and `Pregnancies` (0.22).

### 3. Data Preprocessing (`StandardScaler`)
- **Feature & Target Split:** 8 feature columns (`x`) and 1 target column `Outcome` (`y`).
- **Data Splitting:** 80% train data (**614 samples**) and 20% test data (**154 samples**) with `random_state=42`.
- **Feature Scaling:** Applies `StandardScaler()` to standardize all medical features for optimal KNN performance.

### 4. KNN Training: 4 Distance Metrics x 3 K Variations
This project thoroughly evaluates **KNN** using 4 distance metrics and 3 K values:

| Distance Metric | k=3 | k=5 | k=7 | Best K | Best Accuracy |
|---|---|---|---|---|---|
| **Euclidean** | 70.78% | 69.48% | 68.18% | k=3 | **70.78%** |
| **Manhattan** | 67.53% | 66.23% | 70.78% | k=7 | **70.78%** |
| **Chebyshev** | 61.04% | 68.83% | 67.53% | k=5 | **68.83%** |
| **Minkowski** (p=3) | 68.83% | 70.78% | 67.53% | k=5 | **70.78%** |

### 5. Decision Tree Training
- The **Decision Tree Classifier** with `max_depth=5` achieves an accuracy of **79.22%** ✅, outperforming all KNN variants.
- This model is selected as the **best model** due to its ability to capture non-linear relationships in the medical data.

### 6. Ensemble: Majority Voting
- Combines **5 models** (4 best KNNs + Decision Tree) using `VotingClassifier` with a `hard voting` strategy.
- Ensemble accuracy: **70.13%**, which is lower than the standalone Decision Tree due to the dominance of the less accurate KNN models.

### 7. Model Comparison (Accuracy Ranking)

| Rank | Model | Accuracy |
|---|---|---|
| 🥇 1 | **Decision Tree** | **79.22%** |
| 🥈 2 | KNN Euclidean | 70.78% |
| 🥉 3 | KNN Manhattan | 70.78% |
| 4 | KNN Minkowski | 70.78% |
| 5 | Ensemble (Combined) | 70.13% |
| 6 | KNN Chebyshev | 68.83% |

### 8. Evaluation: Confusion Matrix (Decision Tree)
Based on the best model (Decision Tree) evaluated on the 154 test samples:

|  | Predicted: Healthy (0) | Predicted: Diabetes (1) |
|---|---|---|
| **Actual: Healthy (0)** | ✅ 87 (True Negative) | ❌ 12 (False Positive) |
| **Actual: Diabetes (1)** | ❌ 20 (False Negative) | ✅ 35 (True Positive) |

- **Sensitivity (Recall):** 35 / (35 + 20) = **63.64%**
- **Specificity:** 87 / (87 + 12) = **87.88%**
- **Precision:** 35 / (35 + 12) = **74.47%**

### 9. Simulation: Predicting New Patients
- **Input Data:** `[Pregnancies=2, Glucose=150, BP=75, ST=20, Insulin=100, BMI=33, DPF=0.5, Age=40]`
- **Voting Result:** All **5 algorithms** (4 KNN + Decision Tree) predict **Diabetes (1)**.
- **Conclusion:** The patient is flagged as needing immediate medical attention.

---

### 🎯 What This Project Covers
- Exploratory Data Analysis (EDA) with a correlation heatmap
- Correlation and distribution analysis of medical features
- Binary classification (Healthy / Diabetes)
- Model comparison: KNN (4 distance metrics) vs Decision Tree
- Ensemble Learning with Majority Voting
- Model evaluation with accuracy, confusion matrix, and new patient simulation

---

## 📊 Dataset Overview & Context

This project uses the public **Pima Indians Diabetes** dataset, which contains diagnostic medical data from female patients of Pima Indian heritage aged 21 and older.

- **Source:** [UCI Machine Learning Repository: Pima Indians Diabetes Dataset](https://archive.ics.uci.edu/ml/datasets/diabetes) (Smith et al., 1988).
- **Data Volume:** **768** patient samples with 9 columns.
- **Data Integrity:** Clean dataset with no null values (some features have 0 values representing missing data).

Here is the description of the medical feature columns used to predict the diabetes outcome:

| Feature | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration a 2 hours in an oral glucose tolerance test |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body Mass Index (weight in kg / (height in m)^2) |
| `DiabetesPedigreeFunction` | Diabetes pedigree function (genetic history) |
| `Age` | Age of patient (years) |
| `Outcome` | Class variable: **0** = Healthy, **1** = Diabetes |

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/rapfii/diabetes-knn-dt-ensemble
cd diabetes-knn-dt-ensemble
```

### 2. Create a Virtual Environment *(Recommended)*
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install pandas matplotlib seaborn scikit-learn jupyter
```

---

## 🚀 How to Run

### Run Jupyter Notebook
```bash
jupyter notebook "Diabetes_Prediction_Analysis.ipynb.ipynb"
```

Open your browser and run all cells sequentially. The notebook will:
1. Load and explore the dataset
2. Generate the feature correlation heatmap
3. Train 4 KNN variants + Decision Tree
4. Execute Ensemble (Majority Voting)
5. Show model comparison & confusion matrix
6. Run the new patient prediction simulation

---

## 🧠 ML Workflow

```
Raw Data -> EDA (Correlation Heatmap) -> Feature/Target Split
    -> Train/Test Split (80/20) -> Feature Scaling (StandardScaler)
    -> Training KNN (4 Metrics x 3 K) -> Training Decision Tree
    -> Ensemble Voting -> Evaluation (Confusion Matrix)
    -> New Patient Prediction Simulation
```

---

## 🗂️ Project Structure

```
Source Code & Dataset/
│
├── diabetes.csv                                <- Pima Indians Diabetes Dataset (768 samples)
│
├── Diabetes_Prediction_Analysis.ipynb.ipynb     <- Main notebook: EDA, training, evaluation & prediction
│
├── Output Image/
│   ├── Heatmap Korelasi.png                    <- Correlation heatmap of medical features
│   └── Confusion Matrix DS.png                 <- Confusion matrix for the best model (Decision Tree)
│
└── README.md                                   <- Project documentation (you are here!)
```

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `matplotlib` + `seaborn` | Data visualization (heatmap, confusion matrix) |
| `scikit-learn` (KNeighborsClassifier) | KNN model with 4 distance metrics |
| `scikit-learn` (DecisionTreeClassifier) | Decision Tree model |
| `scikit-learn` (VotingClassifier) | Ensemble Majority Voting |
| `scikit-learn` (StandardScaler) | Feature standardization |
| `scikit-learn` (accuracy_score, confusion_matrix) | Model evaluation |
| `Jupyter Notebook` | Interactive development environment |

---

## 📄 License

This project is licensed under the **MIT License**. Free to use, modify, and distribute.

---

## 🙋 Author

**Rapfi**
- GitHub: https://github.com/rapfii
- LinkedIn: https://www.linkedin.com/in/raffi-khairan-hidayat

---

> ⭐ If you find this project helpful, please give this repository a star!
