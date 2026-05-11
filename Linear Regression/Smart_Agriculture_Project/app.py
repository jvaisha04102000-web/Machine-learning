from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    rainfall = float(request.form["rainfall"])
    soil_quality = float(request.form["soil_quality"])
    temperature = float(request.form["temperature"])
    fertilizer = float(request.form["fertilizer"])
    humidity = float(request.form["humidity"])

    data = np.array([[
        rainfall,
        soil_quality,
        temperature,
        fertilizer,
        humidity
    ]])

    prediction = model.predict(data)[0]

    return render_template(
        "index.html",
        prediction=round(prediction, 2)
    )
if __name__ == "__main__":
    app.run(debug=True)