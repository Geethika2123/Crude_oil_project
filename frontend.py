from flask import Flask, render_template, request, redirect, url_for, session
import pymysql
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from prophet import Prophet
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load models and data
try:
    model_gru = load_model('D:/projectcrude/crude_oil_forecasting/models/gru_residual_model.keras')
    scaler = joblib.load('D:/projectcrude/crude_oil_forecasting/models/residual_scaler.gz')
    df = pd.read_csv('D:\projectcrude\static\Crude oil.csv')
    df['ds'] = pd.to_datetime(df['Date'])
    df['y'] = df['Close/Last']  # Corrected column name to 'Close/Last'
    prophet = Prophet()
    prophet.fit(df[['ds', 'y']])
except Exception as e:
    print(f"Error loading models or data: {str(e)}")
    exit(1)

# MySQL config
def get_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='',  # Adjust password if required
        database='crude_oil_prediction'
    )

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        password_hash = generate_password_hash(password)

        try:
            con = get_connection()
            cursor = con.cursor()
            cursor.execute("INSERT INTO user (name, email, password) VALUES (%s, %s, %s)", (name, email, password_hash))
            con.commit()
            con.close()
            return redirect(url_for('login'))
        except Exception as e:
            return f"Error during registration: {str(e)}"
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            con = get_connection()
            cursor = con.cursor()
            cursor.execute("SELECT * FROM user WHERE email=%s", (email,))
            user = cursor.fetchone()
            con.close()

            if user and check_password_hash(user[3], password):  # assuming password is at index 3
                session['user'] = user[1]
                return redirect(url_for('predict'))
            else:
                return render_template('login.html', error='Invalid email or password')
        except Exception as e:
            return f"Error during login: {str(e)}"
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        date_str = request.form['date']
        start_date = pd.to_datetime(date_str)
        days = int(request.form['days'])

        try:
            results = []
            future = prophet.make_future_dataframe(periods=days, freq='D', include_history=True)  # Include history for trend context
            forecast = prophet.predict(future)

            for i in range(days):
                target_date = start_date + timedelta(days=i)

                # Check if date exists in saved predictions
                saved_results = pd.read_csv("final_hybrid_predictions.csv")
                saved_results['Date'] = pd.to_datetime(saved_results['Date'])

                if target_date in saved_results['Date'].values:
                    row = saved_results[saved_results['Date'] == target_date]
                    prediction = float(row['Hybrid_Prediction_Weighted'].values[0])
                    actual = float(row['Actual'].values[0])
                    error = abs(prediction - actual)
                    results.append({'date': target_date.date(), 'predicted': prediction, 'actual': actual, 'error': error})

                # Check if it's a genuine future date
                elif target_date > df['ds'].max():
                    future_date_index = (target_date - df['ds'].min()).days
                    trend = forecast[forecast['ds'] == target_date]['trend'].values
                    if len(trend) == 0:
                        results.append({'date': target_date.date(), 'predicted': None, 'actual': None, 'error': None,
                                        'message': "Prediction not available for this future date."})
                        continue
                    trend = trend[0]

                    # Get last 60 residuals
                    y_trend = prophet.predict(prophet.make_future_dataframe(periods=0))['trend'].values
                    df['y_trend'] = y_trend
                    df['y_residual'] = df['y'] - df['y_trend']
                    last_60 = df['y_residual'].values[-60:]
                    scaled_input = scaler.transform(last_60.reshape(-1, 1)).reshape(1, 60, 1)

                    predicted_residual_scaled = model_gru.predict(scaled_input)
                    predicted_residual = scaler.inverse_transform(predicted_residual_scaled)[0][0]

                    alpha = 1.05
                    final_prediction = trend + alpha * predicted_residual
                    results.append({'date': target_date.date(), 'predicted': final_prediction, 'actual': None, 'error': None})

                else:
                    results.append({'date': target_date.date(), 'predicted': None, 'actual': None, 'error': None,
                                    'message': "Prediction not available for this past date."})

            return render_template('predict_result.html', start_date=start_date.date(), days=days, results=results, future=(start_date + timedelta(days=days - 1) > df['ds'].max()))

        except Exception as e:
            return f"Error during prediction: {str(e)}"

    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)