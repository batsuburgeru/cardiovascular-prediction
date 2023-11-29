from flask import Flask
from cvdModels import knnPreliminary

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    result = knnPreliminary(data['age'], data['gender'], data['restbps'], data['history'], data['cp'])
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug = True)