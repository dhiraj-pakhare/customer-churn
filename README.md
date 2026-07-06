# Customer Churn Prediction Model

## 📌 Problem
Telecom company loses 27% of customers annually. Predicting churn enables proactive retention.

## 📊 Dataset
- **10,000** telecom customers
- **21** features (contract type, tenure, charges, services)
- **27%** churn rate (imbalanced)

## 🔧 Solution
- **SMOTE** to handle class imbalance
- **XGBoost** model: 87% AUC, 82% Precision, 71% Recall
- **SHAP** for explainability: identifies top churn drivers per customer
- **Streamlit app** for real-time predictions + retention recommendations

## 📈 Key Findings
1. **Contract type** is #1 churn driver (month-to-month = 42% churn)
2. **Fiber optic customers** churn 42% vs 20% for DSL (service quality issue?)
3. **Online security service** reduces churn by 25%

## 🚀 Deployment
Live app: [Streamlit Link](https://dhiraj-customer-churn.streamlit.app)

## 📁 Structure
- `notebooks/` — Full analysis pipeline
- `models/` — Trained XGBoost model
- `app.py` — Streamlit deployment
- `data/` — Raw & processed data

## 📊 Results
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------| 
| XGBoost | 82% | 82% | 71% | 76% | **87%** |
| Random Forest | 80% | 80% | 65% | 72% | 84% |
| Logistic Regression | 78% | 75% | 58% | 66% | 81% |

---

## ✅ How to Use
1. Clone repo
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
4. Input customer details → Get churn prediction + retention strategy

---

**Built with:** Python, XGBoost, SHAP, Streamlit
