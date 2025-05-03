import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import io
import yfinance as yf
import datetime
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_message

# Apply black and classy theme with finance-inspired tweaks
st.set_page_config(page_title="Stock ML Pipeline", layout="wide", page_icon="📈")
st.markdown("""
    <style>
    /* Main background with subtle gradient */
    .stApp {
        background: linear-gradient(to bottom, #1a1a1a, #2a2a2a);
        color: #e0e0e0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2a2a2a;
        color: #e0e0e0;
    }
    
    /* Headers with a finance vibe */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-family: 'Arial', sans-serif;
        text-shadow: 0 0 5px rgba(98, 0, 234, 0.5);
    }
    
    /* Buttons with hover effects */
    .stButton>button {
        background-color: #3a3a3a;
        color: #ffffff;
        border: 1px solid #555555;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4a4a4a;
        border-color: #777777;
        box-shadow: 0 0 10px rgba(98, 0, 234, 0.5);
    }
    
    /* Primary button with finance purple */
    .stButton>button[kind="primary"] {
        background-color: #6200ea;
        border-color: #6200ea;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #7c4dff;
        border-color: #7c4dff;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: #2a2a2a;
        border: 1px solid #444444;
        border-radius: 5px;
    }
    
    /* Metric styling */
    .stMetric {
        background-color: #2a2a2a;
        border: 1px solid #444444;
        border-radius: 5px;
        padding: 10px;
    }
    
    /* Expander styling */
    .stExpander {
        background-color: #2a2a2a;
        border: 1px solid #444444;
        border-radius: 5px;
    }
    
    /* Selectbox and multiselect */
    .stSelectbox, .stMultiSelect {
        background-color: #2a2a2a;
        color: #e0e0e0;
        border: 1px solid #444444;
        border-radius: 5px;
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: #2a2a2a;
        border: 1px solid #444444;
        border-radius: 5px;
    }
    
    /* Plot styling */
    .stPlotlyChart, .stPyplot {
        background-color: #2a2a2a;
        border: 1px solid #444444;
        border-radius: 5px;
        padding: 10px;
    }
    
    /* Input fields */
    .stNumberInput, .stSlider {
        background-color: #2a2a2a;
        color: #e0e0e0;
        border: 1px solid #444444;
        border-radius: 5px;
    }
    
    /* Text and info boxes */
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 5px;
        border: 1px solid;
    }
    .stInfo {
        background-color: #263238;
        border-color: #4f6b75;
        color: #b0bec5;
    }
    .stSuccess {
        background-color: #1b3a2f;
        border-color: #2e7d32;
        color: #a5d6a7;
    }
    .stWarning {
        background-color: #4e342e;
        border-color: #d81b60;
        color: #ffccbc;
    }
    .stError {
        background-color: #3e2723;
        border-color: #d32f2f;
        color: #ef9a9a;
    }
    
    /* Download button */
    .stDownloadButton>button {
        background-color: #3a3a3a;
        color: #ffffff;
        border: 1px solid #555555;
        border-radius: 5px;
    }
    .stDownloadButton>button:hover {
        background-color: #4a4a4a;
        border-color: #777777;
    }
    
    /* Centered GIF styling */
    .center-gif {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = {
            'current_step': 0,
            'data_loaded': False,
            'preprocessed': False,
            'features_engineered': False,
            'data_split': False,
            'model_trained': False,
            'model_evaluated': False,
            'results_visualized': False,
            'df': None,
            'df_processed': None,
            'target': None,
            'features': None,
            'X_train': None,
            'X_test': None,
            'y_train': None,
            'y_test': None,
            'models': {},
            'y_preds': {},
            'current_price': None,
            'last_symbol': None
        }

init_session_state()

# Welcome Interface
def welcome_step():
    st.header("Welcome to the Stock ML Pipeline! 🚀")
    st.markdown("""
        This application guides you through a machine learning pipeline to analyze financial data.  
        Fetch real-time and historical stock data, preprocess, train models, and visualize results with advanced features!
    """)
    
    # Finance-themed GIF (stock market growth chart animation)
    st.markdown("""
        <div class="center-gif">
            <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExMDNpeG53czA0Zm1veXM5OHpma250aTZpNDRzNWZxN3Yydzllc3BwdyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/jBh6MxLLsH9ZNfYLY9/giphy.gif" alt="Stock Market GIF" width="300"/>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("Get Started"):
        st.session_state.pipeline['current_step'] = 1
        st.rerun()

# Helper function to clean numeric columns
def clean_numeric_columns(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception as e:
                st.warning(f"Could not convert column {col} to numeric: {str(e)}")
    return df

# Helper function to check if a series is continuous or categorical
def is_continuous(series):
    if pd.api.types.is_numeric_dtype(series):
        unique_values = len(series.unique())
        return unique_values > 10
    return False

# Helper function to fetch data from yfinance with retry logic and caching
@st.cache_data
def fetch_yfinance_data(symbol, start_date, end_date, _cache_key=None):
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_message(match='Too Many Requests')
    )
    def fetch():
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                st.error(f"No data found for symbol {symbol} in the specified date range. Suggested symbols: AAPL, TSLA, MSFT.")
                return None
            
            df = df.reset_index()
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            return df
        except Exception as e:
            st.error(f"Error fetching data from yfinance: {str(e)}. Suggested symbols: AAPL, TSLA, MSFT.")
            return None
    
    return fetch()

# Helper function to fetch current price
def fetch_current_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        current_data = stock.info
        current_price = current_data.get('regularMarketPrice', current_data.get('currentPrice'))
        if current_price is None:
            return None
        return current_price
    except Exception as e:
        st.warning(f"Could not fetch current price for {symbol}: {str(e)}")
        return None

# Step 1: Load Data
def load_data_step():
    st.header("Step 1: Load Data 📊")
    
    data_option = st.radio("Select data source:", ("Upload your own data", "Fetch data from yfinance"))
    
    if data_option == "Upload your own data":
        uploaded_file = st.file_uploader("Upload your stock data (CSV or Excel)", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                df = clean_numeric_columns(df)
                
                st.session_state.pipeline['df'] = df
                st.session_state.pipeline['data_loaded'] = True
                st.session_state.pipeline['last_symbol'] = None
                
                st.success("✅ Data loaded successfully!")
                
                with st.expander("View Raw Data"):
                    st.dataframe(df)
                
                with st.expander("Data Information"):
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    st.text(buffer.getvalue())
                    st.write("Descriptive Statistics:")
                    st.dataframe(df.describe())
                
                st.session_state.pipeline['current_step'] = 2
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
        else:
            st.info("ℹ️ Please upload a file to get started.")
    
    else:
        st.subheader("Fetch Data from yfinance")
        st.markdown("Enter the stock symbol and date range to fetch data from yfinance.")
        
        symbol = st.text_input("Enter stock symbol (e.g., AAPL)", value="AAPL")
        start_date = st.date_input("Start Date", value=datetime.date(2024, 1, 1))
        end_date = st.date_input("End Date", value=datetime.date(2024, 12, 31))
        
        if st.button("Fetch Data"):
            if symbol and start_date and end_date:
                with st.spinner("Fetching data from yfinance..."):
                    cache_key = f"{symbol}_{start_date}_{end_date}"
                    df = fetch_yfinance_data(symbol.upper(), start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), _cache_key=cache_key)
                
                if df is not None and not df.empty:
                    current_price = fetch_current_price(symbol.upper())
                    if current_price:
                        st.success(f"Current Price of {symbol.upper()}: ${current_price:.2f}")
                        st.session_state.pipeline['current_price'] = current_price
                    
                    st.session_state.pipeline['df'] = df
                    st.session_state.pipeline['data_loaded'] = True
                    st.session_state.pipeline['last_symbol'] = symbol.upper()
                    
                    st.success("✅ Data fetched successfully from yfinance!")
                    
                    with st.expander("View Raw Data"):
                        st.dataframe(df)
                    
                    with st.expander("Data Information"):
                        buffer = io.StringIO()
                        df.info(buf=buffer)
                        st.text(buffer.getvalue())
                        st.write("Descriptive Statistics:")
                        st.dataframe(df.describe())
                    
                    st.session_state.pipeline['current_step'] = 2
                    st.rerun()
            else:
                st.warning("Please provide a stock symbol and date range.")

# Step 2: Preprocessing
def preprocessing_step():
    st.header("Step 2: Preprocessing 🛠️")
    
    if not st.session_state.pipeline['data_loaded']:
        st.warning("Please load data first!")
        return
    
    df = st.session_state.pipeline['df'].copy()
    
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        st.dataframe(missing_values[missing_values > 0].to_frame(name="Missing Count"))
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        st.success("Missing values imputed with mean values")
    else:
        st.success("No missing values found")
    
    st.session_state.pipeline['df_processed'] = df
    st.session_state.pipeline['preprocessed'] = True
    
    with st.expander("View Processed Data"):
        st.dataframe(df)
    
    if st.button("Continue to Feature Engineering"):
        st.session_state.pipeline['current_step'] = 3
        st.rerun()

# Step 3: Feature Engineering
def feature_engineering_step():
    st.header("Step 3: Feature Engineering 📐")
    
    if not st.session_state.pipeline['preprocessed']:
        st.warning("Please complete preprocessing first!")
        return
    
    if st.session_state.pipeline['df_processed'] is None:
        st.error("No processed data found!")
        return
    
    df = st.session_state.pipeline['df_processed'].copy()
    
    st.subheader("Advanced Feature Engineering")
    if 'Close' in df.columns:
        window = st.slider("Select Moving Average window (days)", 5, 50, 20)
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'MA_{window}'] = df[f'MA_{window}'].fillna(df['Close'])
        st.success(f"Added {window}-day Moving Average as a feature!")
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols:
        st.error("No numeric columns found for analysis!")
        return
    
    st.subheader("Feature Selection")
    target = st.selectbox("Select target variable (y)", numeric_cols)
    features = st.multiselect("Select feature variables (X)", [c for c in numeric_cols if c != target])
    
    if not features:
        st.warning("Please select at least one feature!")
        return
    
    st.subheader("Feature Scaling")
    scale_features = st.checkbox("Scale features (Standardization)", value=True)
    
    if scale_features:
        try:
            scaler = StandardScaler()
            df[features] = scaler.fit_transform(df[features])
            st.success("Features successfully scaled!")
        except Exception as e:
            st.error(f"Error scaling features: {str(e)}")
    
    st.subheader("Feature Correlation")
    try:
        corr_matrix = df[features + [target]].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title='Feature Correlation Matrix',
            width=600,
            height=500
        )
        fig.update_layout(
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#1a1a1a',
            font_color='#e0e0e0'
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Could not create correlation matrix: {str(e)}")
    
    st.session_state.pipeline['target'] = target
    st.session_state.pipeline['features'] = features
    st.session_state.pipeline['df_features'] = df
    st.session_state.pipeline['features_engineered'] = True
    
    if st.button("Continue to Train/Test Split"):
        st.session_state.pipeline['current_step'] = 4
        st.rerun()

# Step 4: Train/Test Split
def train_test_split_step():
    st.header("Step 4: Train/Test Split ✂️")
    
    if not st.session_state.pipeline['features_engineered']:
        st.warning("Please complete feature engineering first!")
        return
    
    if st.session_state.pipeline['target'] is None or st.session_state.pipeline['features'] is None:
        st.error("Target or features not selected!")
        return
    
    df = st.session_state.pipeline['df_features']
    target = st.session_state.pipeline['target']
    features = st.session_state.pipeline['features']
    
    st.subheader("Split Configuration")
    test_size = st.slider("Test set size (%)", 10, 40, 20)
    random_state = st.number_input("Random state", 0, 100, 42)
    
    try:
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size/100, 
            random_state=random_state
        )
        
        st.session_state.pipeline['X_train'] = X_train
        st.session_state.pipeline['X_test'] = X_test
        st.session_state.pipeline['y_train'] = y_train
        st.session_state.pipeline['y_test'] = y_test
        st.session_state.pipeline['data_split'] = True
        
        st.subheader("Data Split Visualization")
        split_df = pd.DataFrame({
            'Set': ['Training', 'Testing'],
            'Size': [len(X_train), len(X_test)]
        })
        fig = px.pie(
            split_df,
            names='Set',
            values='Size',
            title='Training vs Testing Split',
            width=400,
            height=400,
            color_discrete_sequence=['#1E90FF', '#FF4500']
        )
        fig.update_layout(
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#1a1a1a',
            font_color='#e0e0e0'
        )
        st.plotly_chart(fig)
        
        st.success("Train/test split completed!")
        
        if st.button("Continue to Model Training"):
            st.session_state.pipeline['current_step'] = 5
            st.rerun()
            
    except Exception as e:
        st.error(f"Error during train/test split: {str(e)}")

# Step 5: Model Training
def model_training_step():
    st.header("Step 5: Model Training 🤖")
    
    if not st.session_state.pipeline['data_split']:
        st.warning("Please complete train/test split first!")
        return
    
    X_train = st.session_state.pipeline['X_train']
    y_train = st.session_state.pipeline['y_train']
    
    st.subheader("Model Configuration")
    model_types = st.multiselect("Select Models to Train", ["Linear Regression", "Logistic Regression", "K-Means Clustering"], default=["Linear Regression"])
    
    target_is_continuous = is_continuous(y_train)
    
    for model_type in model_types:
        if model_type == "Linear Regression" and not target_is_continuous:
            st.warning("""
                ⚠️ Linear Regression expects a continuous target variable (e.g., stock prices). 
                Your target variable appears to be categorical. Consider discretizing it in preprocessing 
                or selecting a different model like Logistic Regression for classification tasks.
            """)
            return
        elif model_type == "Logistic Regression" and target_is_continuous:
            st.warning("""
                ⚠️ Logistic Regression expects a categorical target variable (e.g., buy/sell, 0/1). 
                Your target variable appears to be continuous. You can discretize it in the preprocessing step 
                (e.g., convert to categories like 'high'/'low') or select a different model like Linear Regression.
            """)
            return
    
    models = {}
    for model_type in model_types:
        if model_type == "Linear Regression":
            models[model_type] = LinearRegression()
        elif model_type == "Logistic Regression":
            models[model_type] = LogisticRegression(max_iter=1000)
        else:
            n_clusters = st.number_input("Number of clusters", min_value=2, max_value=10, value=3, key="clusters")
            models[model_type] = KMeans(n_clusters=n_clusters, random_state=42)
    
    with st.spinner("Training models..."):
        try:
            for model_type, model in models.items():
                if model_type in ["Linear Regression", "Logistic Regression"]:
                    model.fit(X_train, y_train)
                else:
                    model.fit(X_train)
                models[model_type] = model
            
            st.session_state.pipeline['models'] = models
            st.session_state.pipeline['model_trained'] = True
            st.success("Model training completed!")
            
            st.subheader("Model Details")
            for model_type, model in models.items():
                st.write(f"**{model_type}**")
                if model_type in ["Linear Regression", "Logistic Regression"]:
                    coef_df = pd.DataFrame({
                        'Feature': ['Intercept'] + st.session_state.pipeline['features'],
                        'Coefficient': [model.intercept_] + list(model.coef_)
                    })
                    st.dataframe(coef_df)
                else:
                    cluster_centers = pd.DataFrame(model.cluster_centers_, columns=st.session_state.pipeline['features'])
                    st.dataframe(cluster_centers)
            
            if st.button("Continue to Evaluation"):
                st.session_state.pipeline['current_step'] = 6
                st.rerun()
                
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")

# Step 6: Evaluation
def evaluation_step():
    st.header("Step 6: Model Evaluation 📊")
    
    if not st.session_state.pipeline['model_trained']:
        st.warning("Please train the model first!")
        return
    
    models = st.session_state.pipeline['models']
    X_test = st.session_state.pipeline['X_test']
    y_test = st.session_state.pipeline['y_test']
    
    y_preds = {}
    try:
        for model_type, model in models.items():
            y_pred = model.predict(X_test)
            y_preds[model_type] = y_pred
        
        st.session_state.pipeline['y_preds'] = y_preds
        
        st.subheader("Model Performance Metrics")
        metrics_df = pd.DataFrame(columns=['Model', 'RMSE', 'R²'])
        for model_type, y_pred in y_preds.items():
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            metrics_df = pd.concat([metrics_df, pd.DataFrame({
                'Model': [model_type],
                'RMSE': [rmse],
                'R²': [r2]
            })], ignore_index=True)
        
        st.dataframe(metrics_df.style.format({'RMSE': '{:.4f}', 'R²': '{:.4f}'}))
        
        st.subheader("Actual vs Predicted Values")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test,
            y=y_test,
            mode='lines',
            name='Ideal Fit',
            line=dict(color='black', dash='dash')
        ))
        for model_type, y_pred in y_preds.items():
            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                name=f'{model_type} Predictions'
            ))
        fig.update_layout(
            title='Actual vs Predicted Values',
            xaxis_title='Actual',
            yaxis_title='Predicted',
            width=600,
            height=400,
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#1a1a1a',
            font_color='#e0e0e0'
        )
        st.plotly_chart(fig)
        
        st.subheader("Residual Plots")
        for model_type, y_pred in y_preds.items():
            residuals = y_test - y_pred
            residual_df = pd.DataFrame({
                'Predicted': y_pred,
                'Residuals': residuals
            })
            fig = px.scatter(
                residual_df,
                x='Predicted',
                y='Residuals',
                title=f'Residual Plot - {model_type}',
                width=600,
                height=400,
                color_discrete_sequence=['#1E90FF']
            )
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            fig.update_layout(
                paper_bgcolor='#1a1a1a',
                plot_bgcolor='#1a1a1a',
                font_color='#e0e0e0'
            )
            st.plotly_chart(fig)
        
        st.session_state.pipeline['model_evaluated'] = True
        
        if st.button("Continue to Results Visualization"):
            st.session_state.pipeline['current_step'] = 7
            st.rerun()
            
    except Exception as e:
        st.error(f"Error during evaluation: {str(e)}")

# Step 7: Results Visualization
def results_visualization_step():
    st.header("Step 7: Results Visualization 📈")
    
    if not st.session_state.pipeline['model_evaluated']:
        st.warning("Please complete model evaluation first!")
        return
    
    df = st.session_state.pipeline['df']
    target = st.session_state.pipeline['target']
    features = st.session_state.pipeline['features']
    y_test = st.session_state.pipeline['y_test']
    y_preds = st.session_state.pipeline['y_preds']
    
    st.subheader("Final Model Performance")
    
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if date_cols:
        date_col = date_cols[0]
        st.subheader("Actual vs Predicted Over Time")
        results_df = pd.DataFrame({'Date': st.session_state.pipeline['X_test'].index})
        results_df['Actual'] = y_test
        for model_type, y_pred in y_preds.items():
            results_df[f'Predicted_{model_type}'] = y_pred
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['Date'],
            y=results_df['Actual'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='#FF4500')
        ))
        for model_type in y_preds.keys():
            fig.add_trace(go.Scatter(
                x=results_df['Date'],
                y=results_df[f'Predicted_{model_type}'],
                mode='lines+markers',
                name=f'Predicted ({model_type})'
            ))
        fig.update_layout(
            title=f'Actual vs Predicted {target} Over Time',
            xaxis_title='Date',
            yaxis_title=target,
            width=800,
            height=400,
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#1a1a1a',
            font_color='#e0e0e0'
        )
        st.plotly_chart(fig)
    else:
        st.info("No date column found for time series visualization")
    
    st.subheader("Feature Importance")
    for model_type, model in st.session_state.pipeline['models'].items():
        if hasattr(model, 'coef_'):
            importance = pd.DataFrame({
                'Feature': features,
                'Importance': model.coef_
            }).sort_values('Importance', ascending=False)
            fig = px.bar(
                importance,
                x='Importance',
                y='Feature',
                title=f'Feature Importance ({model_type})',
                width=800,
                height=400,
                color_discrete_sequence=['#1E90FF']
            )
            fig.update_layout(
                paper_bgcolor='#1a1a1a',
                plot_bgcolor='#1a1a1a',
                font_color='#e0e0e0'
            )
            st.plotly_chart(fig)
    
    st.subheader("Prediction Distribution")
    hist_df = pd.DataFrame({'Value': y_test, 'Type': ['Actual'] * len(y_test)})
    for model_type, y_pred in y_preds.items():
        temp_df = pd.DataFrame({'Value': y_pred, 'Type': [f'Predicted ({model_type})'] * len(y_pred)})
        hist_df = pd.concat([hist_df, temp_df], ignore_index=True)
    
    fig = px.histogram(
        hist_df,
        x='Value',
        color='Type',
        barmode='overlay',
        nbins=30,
        title='Distribution of Predicted vs Actual Values',
        width=800,
        height=400,
        color_discrete_map={'Actual': '#FF4500'},
        opacity=0.7
    )
    fig.update_layout(
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font_color='#e0e0e0',
        xaxis_title='Close/Last',
        yaxis_title='Count'
    )
    st.plotly_chart(fig)
    
    st.subheader("Download Results")
    for model_type, y_pred in y_preds.items():
        results_df = pd.DataFrame({
            'Actual': y_test,
            f'Predicted_{model_type}': y_pred,
            f'Residual_{model_type}': y_test - y_pred
        })
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download Predictions ({model_type}) as CSV",
            data=csv,
            file_name=f'stock_predictions_{model_type.lower().replace(" ", "_")}.csv',
            mime='text/csv',
            key=f"download_{model_type}"
        )
    
    st.session_state.pipeline['results_visualized'] = True
    st.success("Pipeline completed successfully!")
    
    st.markdown("""
        <div class="center-gif">
            <img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExMms0b25xYmJrODE3MGxlMGh0Y29jbm1oZzl1a2RrM28wd2p2bGRwMiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/M9USyermmmO8WJG6FK/giphy.gif" alt="Success GIF" width="300"/>
        </div>
    """, unsafe_allow_html=True)

# Main app layout
def main():
    with st.sidebar:
        st.header("Pipeline Steps")
        
        if st.button("📋 0. Welcome", key="nav0"):
            st.session_state.pipeline['current_step'] = 0
            st.rerun()
        
        if st.button("📊 1. Load Data", key="nav1"):
            st.session_state.pipeline['current_step'] = 1
            st.rerun()
        
        if st.button("🛠️ 2. Preprocessing", 
                    disabled=not st.session_state.pipeline['data_loaded'],
                    key="nav2"):
            st.session_state.pipeline['current_step'] = 2
            st.rerun()
        
        if st.button("📐 3. Feature Engineering", 
                    disabled=not st.session_state.pipeline['preprocessed'],
                    key="nav3"):
            st.session_state.pipeline['current_step'] = 3
            st.rerun()
        
        if st.button("✂️ 4. Train/Test Split", 
                    disabled=not st.session_state.pipeline['features_engineered'],
                    key="nav4"):
            st.session_state.pipeline['current_step'] = 4
            st.rerun()
        
        if st.button("🤖 5. Model Training", 
                    disabled=not st.session_state.pipeline['data_split'],
                    key="nav5"):
            st.session_state.pipeline['current_step'] = 5
            st.rerun()
        
        if st.button("📊 6. Evaluation", 
                    disabled=not st.session_state.pipeline['model_trained'],
                    key="nav6"):
            st.session_state.pipeline['current_step'] = 6
            st.rerun()
            
        if st.button("📈 7. Results Visualization", 
                    disabled=not st.session_state.pipeline['model_evaluated'],
                    key="nav7"):
            st.session_state.pipeline['current_step'] = 7
            st.rerun()
        
        st.divider()
        if st.button("🔄 Reset Pipeline", type="primary"):
            init_session_state()
            st.rerun()
    
    current_step = st.session_state.pipeline['current_step']
    
    if current_step == 0:
        welcome_step()
    elif current_step == 1:
        load_data_step()
    elif current_step == 2:
        preprocessing_step()
    elif current_step == 3:
        feature_engineering_step()
    elif current_step == 4:
        train_test_split_step()
    elif current_step == 5:
        model_training_step()
    elif current_step == 6:
        evaluation_step()
    elif current_step == 7:
        results_visualization_step()

if __name__ == "__main__":
    main()
