import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.serving import run_simple
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple
import pickle


app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("transform.pkl", "rb"))

@app.route('/')
def home():
    
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    
    if request.method == "POST":
        message = request.form["message"]
        data = [message]
        vect = cv.transform(data).toarray()
        pred = model.predict(vect)
    
    return render_template("result.html", prediction=pred)

if __name__ == "__main__":
    run_simple('localhost', 9000, app)