from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Path ke model
MODEL_PATH = os.path.join('model', 'extrovert_introvert_model.pkl')

# Load model dan encoders
data = joblib.load(MODEL_PATH)
model = data['model']
le = data['encoders']['Personality']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari form
        time_spent = float(request.form['Time_spent_Alone'])
        stage_fear = request.form['Stage_fear']  # 'Yes' / 'No'
        social_event = float(request.form['Social_event_attendance'])
        going_outside = float(request.form['Going_outside'])
        drained = request.form['Drained_after_socializing']  # 'Yes' / 'No'
        friends = float(request.form['Friends_circle_size'])
        post_freq = float(request.form['Post_frequency'])

        # Encode fitur kategorikal (Yes=1, No=0)
        stage_fear_encoded = 1 if stage_fear == 'Yes' else 0
        drained_encoded = 1 if drained == 'Yes' else 0

        # Buat array fitur untuk prediksi
        features = np.array([[time_spent, stage_fear_encoded, social_event,
                              going_outside, drained_encoded, friends, post_freq]])

        # Prediksi
        y_pred = model.predict(features)

        # Konversi hasil prediksi ke label asli
        personality = le.inverse_transform(y_pred)[0]

        return render_template('index.html', prediction_text=f"Kamu Seorang {personality}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Terjadi kesalahan: {e}")

if __name__ == '__main__':
    app.run(debug=True)
