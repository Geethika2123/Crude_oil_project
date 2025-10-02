# Crude Oil Price Forecasting System

This project is a **machine learning and deep learning based web application** that forecasts crude oil prices using GRU (Gated Recurrent Unit) models and hybrid prediction approaches. The system integrates a Python backend with a lightweight frontend to provide interactive forecasts, user authentication, and visualization of predictions.

---

## Features

* **Crude Oil Forecasting** using trained GRU-based deep learning models.
* **Hybrid Prediction Models** for improved accuracy.
* **Web Application** with frontend (HTML, CSS, Python) and backend (Flask/Django-based Python scripts).
* **User Authentication**: login, register, and logout functionality.
* **Interactive Prediction Page**: upload/input data and view forecast results.
* **Visualization Support** with charts and tabular predictions.
* **Pre-trained Models** included for immediate use.

---

## Tech Stack

* **Programming Language**: Python
* **Deep Learning**: Keras, TensorFlow
* **Data Handling**: Pandas, NumPy
* **Web Framework**: Flask/Django (check `backend.py`)
* **Frontend**: HTML, CSS
* **Visualization**: Matplotlib, Seaborn (if used)

##  Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/crude-oil-forecasting.git
   cd crude-oil-forecasting/projectcrude
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   *(Create `requirements.txt` with Flask, TensorFlow, Pandas, NumPy, etc.)*

4. **Run the backend server:**

   ```bash
   python backend.py
   ```

5. **Access the app** in your browser at:

   ```
   http://127.0.0.1:5000
   ```

---

## Dataset

* The dataset `Crude oil.csv` contains historical crude oil price data.
* Used for training, testing, and validating forecasting models.

---

##  Machine Learning Models

* **GRU Model**: Time-series forecasting with residual connections.
* **Hybrid Approach**: Combines statistical and deep learning models for improved accuracy.
* Pre-trained models (`.keras`) are included for direct use.

---

## Screenshots

* **Login Page**
<img width="1350" height="719" alt="image" src="https://github.com/user-attachments/assets/e1b4a4a6-cc45-4119-8dff-4357bbd4b08f" />

* **Prediction Input Page**
<img width="1350" height="815" alt="image" src="https://github.com/user-attachments/assets/82cf2785-9b1f-4316-9d22-55495f989dca" />

* **Prediction Results Visualization**
<img width="1350" height="822" alt="image" src="https://github.com/user-attachments/assets/eb327bbf-370e-4ef5-816c-bd9d2beae601" />


## Future Enhancements

* Deployment on cloud (AWS/Heroku).
* API endpoints for external access.
* Advanced visualization dashboards.

---

## Documentation

* Project Report: `Report.pdf`
* Presentation Slides: `PPT.pptx`

---

