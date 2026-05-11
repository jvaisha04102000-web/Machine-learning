from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    distance = float(request.form["distance"])
    traffic = float(request.form["traffic"])
    time_hours = float(request.form["time_hours"])
    weather = float(request.form["weather"])
    peak_hour = float(request.form["peak_hour"])

    data = np.array([[
        distance,
        traffic,
        time_hours,
        weather,
        peak_hour
    ]])

    prediction = model.predict(data)[0]

    return render_template(
        "index.html",
        prediction=round(prediction, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)