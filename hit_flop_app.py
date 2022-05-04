import os
import pickle
from flask import Flask, jsonify, request

import mysklearn.myutils as myutils

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "<h1>Welcome to my app!</h1>", 200

@app.route("/predict", methods=["GET"])
def predict():
    danceability = request.args.get("danceability", "")
    energy = request.args.get("energy", "")
    valence = request.args.get("valence", "")
    tempo = request.args.get("tempo", "")

    prediction = predict_hit_or_flop([danceability, energy, valence, tempo])

    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    return "Error making prediction", 400

def predict_hit_or_flop(instance):
    infile = open("nb.p", "rb")
    nb_classifier, min_tempo, max_tempo = pickle.load(infile)
    infile.close()

    adjusted_instance = ([[myutils.discretize_value(float(instance[0])),
                        myutils.discretize_value(float(instance[1])),
                        myutils.discretize_value(float(instance[2])),
                        myutils.discretize_value(myutils.normalize_tempo_value(float(instance[3]), min_tempo, max_tempo))]])
    
    predicted = nb_classifier.predict(adjusted_instance)

    return predicted[0]

if __name__ == "__main__":
    port = os.environ.get("PORT", 5001)
    app.run(debug=False, port=port, host="0.0.0.0")