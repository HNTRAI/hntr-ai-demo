# HNTR AI – Advisor Intelligence Demo App

This is a Streamlit-based web application demonstrating two core features of the HNTR AI platform:

### 🔥 BLIX™ Score (Breakaway Likelihood Index)
Predicts the likelihood of a financial advisor breaking away from their current firm based on behavioral, engagement, and risk-based signals.

### 🧬 HNTR Fit Score
Clusters top-performing advisors and compares new prospects to assess cultural and performance alignment.

---

## 📦 Features

- CSV upload of advisor data
- Real-time BLIX scoring with risk banding (🔥 Hot / ⚠️ Warm / ❄️ Cold)
- HNTR Fit scoring using clustering + similarity
- Combined score dashboard
- Interactive histograms for score distributions
- Sidebar overview and logo support

---

## 🚀 Getting Started

1. Clone this repo and `cd` into it
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Streamlit app:
```bash
streamlit run hntr_demo_app.py
```
4. Upload a file like `sample_advisors.csv` to get started.

---

## 🧠 About HNTR AI

HNTR AI is a predictive talent intelligence platform that helps broker-dealers and recruiting teams source, engage, and retain financial advisors with unmatched precision and insight.
