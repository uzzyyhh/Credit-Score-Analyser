# 📈 Stock ML Pipeline

A sleek and interactive Streamlit app that guides users through a full machine learning pipeline tailored for stock market data. From data loading to visualization, this app combines modern UI with powerful modeling capabilities and real-time financial data via **Yahoo Finance**.

![Stock ML Pipeline](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExMDNpeG53czA0Zm1veXM5OHpma250aTZpNDRzNWZxN3Yydzllc3BwdyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/jBh6MxLLsH9ZNfYLY9/giphy.gif)

---

## 🚀 Features

- 📥 **Data Loading**  
  - Upload your own CSV/Excel data  
  - Or fetch historical data from Yahoo Finance (via `yfinance`)

- 🧹 **Preprocessing Pipeline**  
  - Auto-detection and conversion of numeric columns  
  - Missing value handling with `SimpleImputer`  
  - Feature scaling with `StandardScaler`

- ⚙️ **Modeling**  
  - Linear Regression, Logistic Regression, and KMeans Clustering  
  - Model training, evaluation, and prediction

- 📊 **Visualization**  
  - Interactive plots via Plotly (line charts, scatter plots, clusters)  
  - Custom metrics and styled output

- 🎨 **Theming**  
  - Finance-inspired dark UI with hover effects and styled components

---

## 📂 Installation

1. Clone the repository:

```
git clone https://github.com/uzzyyhh/stock-ml-pipeline.git
cd stock-ml-pipeline
Install required packages:

pip install -r requirements.txt
Run the Streamlit app:
streamlit run app.py

📦 Dependencies
streamlit

pandas

numpy

plotly

scikit-learn

yfinance

tenacity

Install everything using:
pip install streamlit pandas numpy plotly scikit-learn yfinance tenacity

🧠 How It Works
The app follows a step-by-step pipeline:

Welcome Page – Introduction + stock market themed GIF

Data Load – Upload or fetch from Yahoo Finance

Preprocessing – Clean, impute, and scale data

Modeling – Choose models and train

Evaluation & Visualization – Analyze results and visualize with Plotly

📈 Example Tickers
If using Yahoo Finance, try:

AAPL – Apple Inc.

TSLA – Tesla, Inc.

MSFT – Microsoft Corporation

🛡️ Caching & Resilience
Yahoo Finance requests are cached using @st.cache_data

Auto-retry mechanism using tenacity for handling rate limits and request errors

💡 Future Enhancements
Add deep learning models (LSTM, GRU)

Support for technical indicators (MACD, RSI)

Dashboard export (PDF, Excel)

API integration for crypto and forex

📃 License
MIT License

👨‍💻 Author
Made with ❤️ by Usman Nadeem
Feel free to contribute or fork the repo!
