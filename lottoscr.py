
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import time
import schedule
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import re

# Enable eager execution in TensorFlow
tf.config.run_functions_eagerly(True)

# Configuration
LOTTERY_CONFIG = {
    'Lotto Max': {
        'url': 'https://www.olg.ca/en/lottery/play-lotto-max-encore/past-results.html',
        'csv': 'lottery_results.csv',
        'model': 'modelmax.h5',
        'num_numbers': 8,
        'max_num': 50
    },
    'Lotto 649': {
        'url': 'https://www.olg.ca/en/lottery/play-lotto-649-encore/past-results.html',
        'csv': 'lotto649.csv',
        'model': 'model649.h5',
        'num_numbers': 7,
        'max_num': 49
    }
}

# Custom CSS for number display
st.markdown("""
<style>
.number-card {
    padding: 10px;
    border-radius: 10px;
    margin: 10px;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
}
.number-badge {
    display: inline-block;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: #1E88E5;
    color: white;
    text-align: center;
    line-height: 25px;
    margin: 5px;

}
</style>
""", unsafe_allow_html=True)

# Initialize session state for prediction states
if 'lotto_max_state' not in st.session_state:
    st.session_state.lotto_max_state = 'Tuesday'  # Initial state for Lotto Max
if 'lotto_649_state' not in st.session_state:
    st.session_state.lotto_649_state = 'Wednesday'  # Initial state for Lotto 649


def scrape_lottery(lottery_name):
    config = LOTTERY_CONFIG[lottery_name]
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        # Set up ChromeDriver with correct permissions
        chromedriver_path = ChromeDriverManager().install()
        os.chmod(chromedriver_path, 0o755)
        service = Service(chromedriver_path)
        
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(config['url'])
        time.sleep(3)  # Wait for page load
        
        raw_data = driver.find_element(By.CSS_SELECTOR, 
                     "ul.extra-bottom.draw-balls.remove-default-styles.ball-list").text
        driver.quit()
        
        numbers = re.findall(r'\d+', raw_data)[:config['num_numbers']]
        return [int(n) for n in numbers]
    except Exception as e:
        st.error(f"Error scraping {lottery_name}: {str(e)}")
        return None


def update_and_predict(lottery_name):
    """Update data and make predictions"""
    config = LOTTERY_CONFIG[lottery_name]
    
    with st.spinner(f"Updating {lottery_name} data..."):
        new_numbers = scrape_lottery(lottery_name)
        if not new_numbers:
            return False
        
        try:
            df = pd.read_csv(config['csv'])
            if df.iloc[-1, 1:].tolist() == new_numbers:
                st.info("No new numbers found")
                return True
        except:
            df = pd.DataFrame(columns=['Date'] + [f'W{i+1}' for i in range(config['num_numbers'])])
        
        new_row = pd.DataFrame([[datetime.now().date()] + new_numbers], 
                             columns=df.columns)
        df = pd.concat([df, new_row])
        df.to_csv(config['csv'], index=False)
        
    with st.spinner("Retraining model..."):
        try:
            model = tf.keras.models.load_model(config['model'])
            # Recompile the model after loading
            model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
        except:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(config['num_numbers'],)),
                tf.keras.layers.Dense(config['num_numbers'])
            ])
            model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
        
        X = df.iloc[:, 1:].values[:-1]
        y = df.iloc[:, 1:].values[1:]
        model.fit(X/config['max_num'], y/config['max_num'], 
                epochs=50, verbose=0)
        model.save(config['model'])
    
    # Toggle the prediction state
    if lottery_name == 'Lotto Max':
        st.session_state.lotto_max_state = 'Friday' if st.session_state.lotto_max_state == 'Tuesday' else 'Tuesday'
    elif lottery_name == 'Lotto 649':
        st.session_state.lotto_649_state = 'Saturday' if st.session_state.lotto_649_state == 'Wednesday' else 'Wednesday'
    
    return True

def generate_predictions(base_prediction, config):
     """Generate multiple predictions with perturbations"""
     predictions = []
    
     predictions.append(np.clip(base_prediction, 1, config['max_num']))
    
     perturbation = np.random.randint(-2, 3, size=config['num_numbers'])
     better_pred = base_prediction + perturbation
     predictions.append(np.clip(better_pred, 1, config['max_num']))
    
     perturbation = np.random.randint(-3, 4, size=config['num_numbers'])
     good_pred = base_prediction + perturbation
     predictions.append(np.clip(good_pred, 1, config['max_num']))
    
     return predictions

def display_numbers(numbers, title):
    """Display numbers in styled cards"""
    st.markdown(f"<div class='number-card'><h3>{title}</h3>", unsafe_allow_html=True)
    cols = st.columns(len(numbers))
    for col, num in zip(cols, numbers):
        with col:
            st.markdown(f"<div class='number-badge'>{num}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def main():
    st.title("Ontario Lottery Predictor")
    st.markdown("---")
    
    # Create tabs for different lotteries
    tab1, tab2 = st.tabs(["Lotto Max", "Lotto 649"])
    
    with tab1:
        st.header("Lotto Max Predictions")
        if st.button("Update Lotto Max"):
            if update_and_predict('Lotto Max'):
                st.success("Lotto Max updated successfully!")
        
        col1, col2 = st.columns(2)
        with col1:
            try:
                df = pd.read_csv(LOTTERY_CONFIG['Lotto Max']['csv'])
                last_numbers = df.iloc[-1, 1:].tolist()
                display_numbers(last_numbers, "Last Draw Numbers")
            except:
                st.warning("No Lotto Max data available")
        
        with col2:
            try:
                model = tf.keras.models.load_model(LOTTERY_CONFIG['Lotto Max']['model'])
                
                # Ensure there are at least 10 draws to use for prediction
                if len(df) < 10:
                    st.warning("Not enough data to make a prediction")
                    return
                
                X = df.iloc[-1, 1:].values.astype(np.float32).reshape(1, -1) / LOTTERY_CONFIG['Lotto Max']['max_num']
                pred = model.predict(X)[0]
                pred_numbers = np.clip(np.round(pred * LOTTERY_CONFIG['Lotto Max']['max_num']).astype(int), 1, 50)
            
                predictions = generate_predictions(pred_numbers, LOTTERY_CONFIG['Lotto Max'])
                labels = [f"Best     {st.session_state.lotto_max_state} Prediction", 
                          f"Better     {st.session_state.lotto_max_state} Prediction", 
                          f"Good     {st.session_state.lotto_max_state} Prediction"]
            
                for label, pred in zip(labels, predictions):
                     display_numbers(pred, label)
                
            except Exception as e:
                 st.warning(f"Prediction not available: {str(e)}")
    
    with tab2:
        st.header("Lotto 649 Predictions")
        if st.button("Update Lotto 649"):
            if update_and_predict('Lotto 649'):
                st.success("Lotto 649 updated successfully!")
        
        col1, col2 = st.columns(2)
        with col1:
            try:
                df = pd.read_csv(LOTTERY_CONFIG['Lotto 649']['csv'])
                last_numbers = df.iloc[-1, 1:].tolist()
                display_numbers(last_numbers, "Last Draw Numbers")
            except:
                st.warning("No Lotto 649 data available")
        
        with col2:
            try:
                model = tf.keras.models.load_model(LOTTERY_CONFIG['Lotto 649']['model'])
                
                # Ensure there are at least 10 draws to use for prediction
                if len(df) < 10:
                    st.warning("Not enough data to make a prediction")
                    return
                
                X = df.iloc[-1, 1:].values.astype(np.float32).reshape(1, -1) / LOTTERY_CONFIG['Lotto 649']['max_num']
                pred = model.predict(X)[0]
                pred_numbers = np.round(pred * LOTTERY_CONFIG['Lotto 649']['max_num']).astype(int)
                
                predictions = generate_predictions(pred_numbers, LOTTERY_CONFIG['Lotto 649'])
                labels = [f"Best {st.session_state.lotto_649_state}    Prediction", 
                          f"Better {st.session_state.lotto_649_state}   Prediction", 
                          f"Good {st.session_state.lotto_649_state}    Prediction"]
                
                for label, pred in zip(labels, predictions):
                    display_numbers(pred, label)
                    
            except Exception as e:
             st.warning(f"Prediction not available: {str(e)}")
    
    st.markdown("---")
    st.markdown("**Note:** Predictions are based on historical patterns and should not be considered financial advice")

if __name__ == "__main__":
    main()
