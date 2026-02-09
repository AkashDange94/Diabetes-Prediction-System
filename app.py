from flask import Flask, render_template, request
import pickle
import numpy as np
import webbrowser
import threading

app = Flask(__name__)

# Load your trained model (make sure model.pkl is in the same folder)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input values from form
        inputs = [float(request.form[field]) for field in ['Pregnancies', 'Glucose', 'BloodPressure',
                                                          'SkinThickness', 'Insulin', 'BMI',
                                                          'DiabetesPedigreeFunction', 'Age']]
        inputs_array = np.array([inputs])
        result = model.predict(inputs_array)[0]
        prediction = "Diabetic" if result == 1 else "Not Diabetic"
        
        # Advice messages based on prediction
        if prediction == "Diabetic":
            advice = [
                "Maintain a balanced diet low in sugar and refined carbs.",
                "Exercise regularly to manage blood sugar levels.",
                "Monitor your blood glucose frequently.",
                "Consult your healthcare provider for medication and advice.",
                "Avoid smoking and limit alcohol consumption."
            ]
        else:
            advice = [
                "Keep a healthy lifestyle with regular physical activity.",
                "Eat a balanced diet rich in vegetables and whole grains.",
                "Maintain a healthy weight to reduce risk.",
                "Schedule regular check-ups to monitor your health.",
                "Stay hydrated and manage stress effectively."
            ]
        
        return render_template("index.html", prediction=prediction, advice=advice)
    
    except:
        return render_template("index.html", prediction="❌ Invalid input. Please fill all fields correctly.", advice=[])

# Auto open browser when app starts
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(debug=True)
