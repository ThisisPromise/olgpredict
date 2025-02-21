import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import time
import schedule
import os
import re
import requests
from bs4 import BeautifulSoup
import asyncio
import concurrent.futures
from playwright.async_api import async_playwright

# Ensure that the Playwright browsers and dependencies are installed
os.system('playwright install')
os.system('playwright install-deps')

# Enable eager execution for TensorFlow functions
tf.config.run_functions_eagerly(True)

# Lottery configuration for two different games
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

if 'lotto_max_state' not in st.session_state:
    st.session_state.lotto_max_state = 'Tuesday'
if 'lotto_649_state' not in st.session_state:
    st.session_state.lotto_649_state = 'Wednesday'

# --- Asynchronous Playwright scraping function ---
async def async_scrape_lottery(lottery_name):
    config = LOTTERY_CONFIG[lottery_name]
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
            )
            page = await browser.new_page()
            await page.goto(config['url'])
            await page.wait_for_timeout(3000)  # Wait for JavaScript to load content
            
            # Get the text content of the lottery numbers
            element = await page.query_selector("ul.extra-bottom.draw-balls.remove-default-styles.ball-list")
            if not element:
                st.error("Could not locate lottery numbers element on the page.")
                await browser.close()
                return None

            # Preserve spaces while extracting text
            raw_data = await element.inner_text()
            await browser.close()

        # Debugging: Print raw extracted text
        st.write(f"Raw extracted text: {raw_data}")

        # Extract numbers while ensuring correct spacing
        numbers = re.findall(r'\d+', raw_data)

        # Convert to integers
        numbers = [int(n) for n in numbers]

        # Debugging: Print cleaned numbers
        st.write(f"Cleaned extracted numbers: {numbers}")

        # Ensure correct number count
        if len(numbers) != config['num_numbers']:
            st.error(f"Error: Expected {config['num_numbers']} numbers, but got {len(numbers)}")
            return None

        return numbers
    except Exception as e:
        st.error(f"Error scraping {lottery_name}: {str(e)}")
        return None

# Synchronous wrapper for running Playwright async functions
def scrape_lottery(lottery_name):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(lambda: asyncio.run(async_scrape_lottery(lottery_name)))
        return future.result()


def update_and_predict(lottery_name):
    """Update data and make predictions"""
    config = LOTTERY_CONFIG[lottery_name]
    
    with st.spinner(f"Updating {lottery_name} data..."):
        new_numbers = scrape_lottery(lottery_name)
        
        # Debug: Print scraped numbers
        st.write(f"Scraped numbers for {lottery_name}: {new_numbers} (Expected: {config['num_numbers']})")

        if not new_numbers or len(new_numbers) != config['num_numbers']:
            st.error(f"Error: Expected {config['num_numbers']} numbers, but got {len(new_numbers) if new_numbers else 'None'}")
            return False

        try:
            df = pd.read_csv(config['csv'])
            st.write(f"Loaded CSV successfully. Columns: {df.columns.tolist()}")
            
            # Check if the last row already matches the scraped numbers
            if df.iloc[-1, 1:].tolist() == new_numbers:
                st.info("No new numbers found")
                return True
        except Exception as e:
            st.write(f"CSV file not found or error reading CSV: {e}. Creating new DataFrame.")
            df = pd.DataFrame(columns=['Date'] + [f'W{i+1}' for i in range(config['num_numbers'])])

        # **Ensure Correct Data Alignment Before Adding New Row**
        try:
            new_row_data = [datetime.now().date()] + new_numbers
            st.write(f"New row data: {new_row_data} (Expected {len(df.columns)} columns)")

            new_row = pd.DataFrame([new_row_data], columns=df.columns)
        except ValueError as e:
            st.error(f"ValueError while creating new DataFrame row: {e}")
            return False

        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(config['csv'], index=False)

    with st.spinner("Retraining model..."):
        try:
            model = tf.keras.models.load_model(config['model'])
            model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
        except Exception as e:
            st.warning(f"Model loading failed: {e}. Creating a new model.")
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(config['num_numbers'],)),
                tf.keras.layers.Dense(config['num_numbers'])
            ])
            model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

        X = df.iloc[:, 1:].values[:-1]
        y = df.iloc[:, 1:].values[1:]
        model.fit(X / config['max_num'], y / config['max_num'], epochs=50, verbose=0)
        model.save(config['model'])

    if lottery_name == 'Lotto Max':
        st.session_state.lotto_max_state = 'Friday' if st.session_state.lotto_max_state == 'Tuesday' else 'Tuesday'
    elif lottery_name == 'Lotto 649':
        st.session_state.lotto_649_state = 'Saturday' if st.session_state.lotto_649_state == 'Wednesday' else 'Wednesday'

    return True

# --- Prediction generation ---
def generate_predictions(base_prediction, config):
    """Generate multiple predictions with perturbations"""
    predictions = []
    predictions.append(np.clip(base_prediction, 1, config['max_num']))
    perturbation = np.random.randint(-2, 3, size=config['num_numbers'])
    predictions.append(np.clip(base_prediction + perturbation, 1, config['max_num']))
    perturbation = np.random.randint(-3, 4, size=config['num_numbers'])
    predictions.append(np.clip(base_prediction + perturbation, 1, config['max_num']))
    return predictions

# --- Display function ---
def display_numbers(numbers, title):
    """Display numbers in styled cards"""
    st.markdown(f"<div class='number-card'><h3>{title}</h3>", unsafe_allow_html=True)
    cols = st.columns(len(numbers))
    for col, num in zip(cols, numbers):
        with col:
            st.markdown(f"<div class='number-badge'>{num}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Main Streamlit App ---
def main():
    st.title("Ontario Lottery Predictor")
    st.markdown("---")
    
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
            except Exception as e:
                st.warning("No Lotto Max data available")
        with col2:
            try:
                model = tf.keras.models.load_model(LOTTERY_CONFIG['Lotto Max']['model'])
                if len(df) < 10:
                    st.warning("Not enough data to make a prediction")
                    return
                X = df.iloc[-1, 1:].values.astype(np.float32).reshape(1, -1) / LOTTERY_CONFIG['Lotto Max']['max_num']
                pred = model.predict(X)[0]
                pred_numbers = np.clip(np.round(pred * LOTTERY_CONFIG['Lotto Max']['max_num']).astype(int), 1, 50)
                predictions = generate_predictions(pred_numbers, LOTTERY_CONFIG['Lotto Max'])
                labels = [f"Best {st.session_state.lotto_max_state} Prediction",
                          f"Better {st.session_state.lotto_max_state} Prediction",
                          f"Good {st.session_state.lotto_max_state} Prediction"]
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
            except Exception as e:
                st.warning("No Lotto 649 data available")
        with col2:
            try:
                model = tf.keras.models.load_model(LOTTERY_CONFIG['Lotto 649']['model'])
                if len(df) < 10:
                    st.warning("Not enough data to make a prediction")
                    return
                X = df.iloc[-1, 1:].values.astype(np.float32).reshape(1, -1) / LOTTERY_CONFIG['Lotto 649']['max_num']
                pred = model.predict(X)[0]
                pred_numbers = np.clip(np.round(pred * LOTTERY_CONFIG['Lotto 649']['max_num']).astype(int), 1, 49)
                predictions = generate_predictions(pred_numbers, LOTTERY_CONFIG['Lotto 649'])
                labels = [f"Best {st.session_state.lotto_649_state} Prediction",
                          f"Better {st.session_state.lotto_649_state} Prediction",
                          f"Good {st.session_state.lotto_649_state} Prediction"]
                for label, pred in zip(labels, predictions):
                    display_numbers(pred, label)
            except Exception as e:
                st.warning(f"Prediction not available: {str(e)}")
    
    st.markdown("---")
    st.markdown("**Note:** Predictions are based on historical patterns and should not be considered financial advice")



if __name__ == "__main__":
    main()
