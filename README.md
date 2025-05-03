# 📈 Stock ML Pipeline

Welcome to **Stock ML Pipeline**, an interactive Streamlit web application that enables users to perform end-to-end financial stock analysis using real-time or uploaded data. From data loading and preprocessing to model training and results visualization — this app is a complete machine learning pipeline tailored for stock market enthusiasts, data scientists, and finance professionals.

---

## 🚀 Features

- **Modern UI**: Custom dark theme with a classy financial look.
- **Data Loading**:
  - Upload your own dataset (CSV or Excel).
  - Fetch real-time stock data from Yahoo Finance using `yfinance`.
- **Preprocessing**:
  - Automatic detection of numerical/categorical features.
  - Cleaning and imputation.
  - Feature engineering.
- **Machine Learning**:
  - Linear Regression
  - Logistic Regression
  - KMeans Clustering
- **Model Evaluation**:
  - Metrics: RMSE, R² Score
  - Visualizations with Plotly
- **Stock Insights**:
  - Real-time price retrieval
  - Historical trends & volume analysis
- **Robust Retry Logic** for API data fetching using `tenacity`.

---

## 🛠️ Tech Stack

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Plotly](https://plotly.com/)
- [Scikit-learn](https://scikit-learn.org/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [tenacity](https://tenacity.readthedocs.io/)

---

## 📷 Interface Preview

![Stock Market App GIF](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExMDNpeG53czA0Zm1veXM5OHpma250aTZpNDRzNWZxN3Yydzllc3BwdyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/jBh6MxLLsH9ZNfYLY9/giphy.gif)

---

## ▶️ Getting Started

### 1. Clone the Repo

```
git clone https://github.com/uzzyyhh/stock-ml-pipeline.git
cd stock-ml-pipeline
2. Install Dependencies
pip install -r requirements.txt
3. Run the App
streamlit run app.py
📂 File Structure
bash
Copy
Edit
├── app.py                # Main Streamlit application
├── README.md             # Project overview
├── requirements.txt      # Dependencies
└── data/                 # (Optional) Place your sample datasets here
💡 Future Enhancements
Add deep learning models (LSTM, GRU)

Support for sentiment analysis from financial news

Integration with financial APIs (Alpha Vantage, Finnhub)

Model comparison dashboard

🧑‍💻 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change or improve.

📝 License
MIT

📬 Contact
For feedback or collaboration:

📧 Email: i229831@nu.edu.pk

