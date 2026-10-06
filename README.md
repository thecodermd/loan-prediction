# 🏦 Loan Approval Prediction ML Project

A complete, production-ready Machine Learning application that predicts loan approval status using multiple ML algorithms with a stunning Streamlit dashboard.

---

## 🌟 Features

- **Multi-Model Training**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- **Best Model Auto-Selection**: Automatically selects and saves the best performing model
- **Interactive Predictions**: Real-time loan approval predictions with probability scores
- **Explainability**: Feature importance & permutation-based factor analysis
- **EDA Dashboard**: Comprehensive data exploration with interactive Plotly charts
- **Dark Theme UI**: Modern dark-themed Streamlit interface with purple accents
- **Auto-Training**: Automatically trains models on first run if no models are found

---

## 📁 Project Structure

```
loan-approval-app/
├── app.py                  # Main Streamlit application
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .streamlit/
│   └── config.toml         # Streamlit theme & server config
├── data/
│   └── loan_data.csv       # Synthetic loan dataset (auto-generated)
└── models/
    ├── best_model.pkl      # Best trained model
    ├── preprocessor.pkl    # Fitted preprocessor pipeline
    ├── feature_names.pkl   # Feature names list
    ├── metrics.pkl         # Best model metrics
    └── model_comparison.pkl # All model metrics
```

---

## 🚀 Quick Start

### 1. Clone / Download the project

```bash
git clone <your-repo-url>
cd loan-approval-app
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (optional — auto-trains on first run)

```bash
python train_model.py
```

### 5. Launch the Streamlit app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📊 Pages Overview

| Page | Description |
|------|-------------|
| 🏠 Home | Hero section, key metrics, dataset overview |
| 🔮 Predict Loan | Interactive form to predict loan approval |
| 📊 EDA & Insights | Data exploration, distributions, correlations |
| 🤖 Model Performance | Model comparison, ROC curves, confusion matrix |
| 📋 About | Project info, tech stack, how it works |

---

## 🤖 ML Approach

### Dataset
- **5,000 synthetic records** generated with realistic distributions
- Features: Gender, Marital Status, Dependents, Education, Self-Employment, Income, Loan Amount, Loan Term, Credit History, Property Area
- **Credit History** is the strongest predictor (mimics real-world lending)

### Preprocessing
- **Categorical Encoding**: One-Hot Encoding for nominal features
- **Numeric Scaling**: StandardScaler for numerical features
- **Missing Value Handling**: Median imputation for numerics, mode for categoricals
- **Pipeline**: Scikit-learn `ColumnTransformer` + `Pipeline`

### Models Evaluated
1. **Logistic Regression** — Baseline linear model
2. **Random Forest** — Ensemble of decision trees
3. **Gradient Boosting** — Sequential boosting ensemble
4. **XGBoost** — Optimized gradient boosting

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Best model selected based on **ROC-AUC** score

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Frontend | Streamlit 1.28+ |
| ML Framework | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Model Persistence | Joblib |
| Language | Python 3.8+ |

---

## ☁️ Deploy to Streamlit Cloud

1. Push your project to a **public GitHub repository**
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** and connect your GitHub repo
4. Set the main file path to `app.py`
5. Click **Deploy**

> **Note**: The app auto-trains models on first deployment. Ensure `train_model.py` is included.

---

## 📸 Screenshots

> *(Add screenshots of your app here)*

---

## 📄 License

This project is open-source under the MIT License.

---

## 🙋 Contact

- **GitHub**: [your-github-handle](https://github.com/your-handle)
- **Email**: your@email.com

---

*Built with ❤️ using Python, Scikit-learn, XGBoost, and Streamlit*
