import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, redirect, render_template, request, session, url_for
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)
app.secret_key = 'p1234'

model_soc_dnn = tf.keras.models.load_model('model_soc_dnn.keras')
model_soh_dnn = tf.keras.models.load_model('model_soh_dnn.keras')

datasets = {
    "Dataset 1": "00005.csv",
    "Dataset 2": "00006.csv",
    "Dataset 3": "00007.csv",
    "Dataset 4": "00018.csv",
    "Dataset 5": "finalData.csv"
}

model_results = {
    "DNN": {"MSE": 0.01, "R2": 0.95},
    "LSTM": {"MSE": 0.02, "R2": 0.90}
}

def predict_soc(inputs):
    try:
        inputs_array = np.array([inputs])  # Convert input list to a 2D NumPy array
        prediction = model_soc_dnn.predict(inputs_array)[0][0]  # Predict using the SoC model
        print(f"SoC Prediction Input: {inputs}, Prediction: {prediction}")  # Print for debugging
        return prediction
    except Exception as e:
        print(f"Error in SoC prediction: {e}")  # Log the error
        return f"Error: {str(e)}"

def predict_soh(inputs):
    try:
        inputs_array = np.array([inputs])  # Convert input list to a 2D NumPy array
        prediction = model_soh_dnn.predict(inputs_array)[0][0]  # Predict using the SoH model
        print(f"SoH Prediction Input: {inputs}, Prediction: {prediction}")  # Print for debugging
        return prediction
    except Exception as e:
        print(f"Error in SoH prediction: {e}")  # Log the error
        return f"Error: {str(e)}"

def load_users():
    try:
        with open('users.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_users(users):
    with open('users.json', 'w') as f:
        json.dump(users, f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('dashboard'))  # Redirect to the dashboard
        else:
            return "Invalid credentials!"
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        if username in users:
            return "User already exists!"
        users[username] = password
        save_users(users)
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'username' not in session:
        return redirect(url_for('login'))

    soc_prediction = None
    soh_prediction = None
    inputs_soc = None  # For holding inputs for SoC prediction
    inputs_soh = None  # For holding inputs for SoH prediction

    if request.method == 'POST':
        # Handle SoC prediction
        if 'predict_soc' in request.form:
            try:
                # Get inputs with validation for empty values
                voltage_measured = request.form['Voltage_measured']
                current_measured = request.form['Current_measured']
                voltage_charge = request.form['Voltage_charge']

                if not voltage_measured or not current_measured or not voltage_charge:
                    raise ValueError("Please fill all fields for SoC prediction.")

                inputs_soc = {
                    'Voltage_measured': voltage_measured,
                    'Current_measured': current_measured,
                    'Voltage_charge': voltage_charge
                }
                inputs = [
                    float(voltage_measured),
                    float(current_measured),
                    float(voltage_charge)
                ]
                soc_prediction = model_soc_dnn.predict(np.array([inputs]))[0][0]
            except Exception as e:
                soc_prediction = f"Error: {str(e)}"

        # Handle SoH prediction
        elif 'predict_soh' in request.form:
            try:
                # Get inputs with validation for empty values
                capacity = request.form['Capacity']
                capacity_fade = request.form['Capacity_Fade']
                resistance_increase = request.form['Resistance_Increase']
                current_charge = request.form['Current_charge']
                voltage_charge = request.form['Voltage_charge']

                if not capacity or not capacity_fade or not resistance_increase or not current_charge or not voltage_charge:
                    raise ValueError("Please fill all fields for SoH prediction.")

                inputs_soh = {
                    'Capacity': capacity,
                    'Capacity_Fade': capacity_fade,
                    'Resistance_Increase': resistance_increase,
                    'Current_charge': current_charge,
                    'Voltage_charge': voltage_charge
                }
                inputs = [
                    float(capacity),
                    float(capacity_fade),
                    float(resistance_increase),
                    float(current_charge),
                    float(voltage_charge)
                ]
                soh_prediction = model_soh_dnn.predict(np.array([inputs]))[0][0]
            except Exception as e:
                soh_prediction = f"Error: {str(e)}"

    return render_template('prediction.html', 
                           soc_prediction=soc_prediction, 
                           soh_prediction=soh_prediction,
                           inputs_soc=inputs_soc, 
                           inputs_soh=inputs_soh)



@app.route('/view_dataset', methods=['GET', 'POST'])
def view_dataset():
    dataset_name = None
    data = None

    if request.method == 'POST':
        dataset_name = request.form['dataset']
        dataset_file = f"data/{dataset_name}.csv"  # Assuming CSV files are named like 00005.csv, 00006.csv, etc.

        # Check if the file exists
        if os.path.exists(dataset_file):
            data = pd.read_csv(dataset_file).head(10).to_dict(orient='records')  # Read first 10 rows
        else:
            data = None

    return render_template('view_dataset.html', dataset_name=dataset_name, data=data)

@app.route('/model_results')
def model_results_page():
    return render_template('model_results.html', results=model_results)


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
