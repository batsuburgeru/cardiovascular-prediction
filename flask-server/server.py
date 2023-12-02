from flask import Flask, request, jsonify
from cvdModels import knnPreliminary, logisticRegressionPreliminary, svmPreliminary
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app, origins="*")

@app.route('/prelim', methods=['POST'])
def prelim():
    data = request.get_json(force=True)  # get the posted data

    age = data['age']
    gender = data['gender']
    trestbps = data['trestbps']
    history = data['history']
    cp = data['cp']

    lrPrelimPredictedClass, lrPrelimProbability, lrPrelimAccuracy, lrPrelimConfusionMatrix, lrPrelimPrecision, lrPrelimRecall, lrPrelimF1, lrPrelimMse, lrPrelimRmse = logisticRegressionPreliminary(age, gender, trestbps, history, cp)
    knnPrelimPredictedClass, knnPrelimProbability, knnPrelimAccuracy, knnPrelimConfusionMatrix, knnPrelimPrecision, knnPrelimRecall, knnPrelimF1, knnPrelimMse, knnPrelimRmse = knnPreliminary(age, gender, trestbps, history, cp)
    svmPrelimPredictedClass, svmPrelimProbability, svmPrelimAccuracy, svmPrelimConfusionMatrix, svmPrelimPrecision, svmPrelimRecall, svmPrelimF1, svmPrelimMse, svmPrelimRmse = svmPreliminary(age, gender, trestbps, history, cp)


    response = {
        'Logistic Regression Predicted Class': lrPrelimPredictedClass.tolist(),
        'Logistic Regression Probability': lrPrelimProbability,
        'Logistic Regression Accuracy': lrPrelimAccuracy,
        'Logistic Regression Confusion Matrix': lrPrelimConfusionMatrix.tolist(),
        'Logistic Regression Precision': lrPrelimPrecision,
        'Logistic Regression Recall': lrPrelimRecall,
        'Logistic Regression F1 Score': lrPrelimF1,
        'Logistic Regression Mean Squared Error (MSE)': lrPrelimMse,
        'Logistic Regression Root Mean Squared Error (RMSE)': lrPrelimRmse,

        'KNN Predicted Class': knnPrelimPredictedClass.tolist(),
        'KNN Probability':knnPrelimProbability,
        'KNN Accuracy': knnPrelimAccuracy,
        'KNN Confusion Matrix': knnPrelimConfusionMatrix.tolist(),
        'KNN Precision': knnPrelimPrecision,
        'KNN Recall': knnPrelimRecall,
        'KNN F1 Score': knnPrelimF1,
        'KNN Mean Squared Error (MSE)': knnPrelimMse,
        'KNN Root Mean Squared Error (RMSE)': knnPrelimRmse,

        'SVM Predicted Class': svmPrelimPredictedClass.tolist(),
        'SVM Probability':svmPrelimProbability,
        'SVM Accuracy': svmPrelimAccuracy,
        'SVM Confusion Matrix': svmPrelimConfusionMatrix.tolist(),
        'SVM Precision': svmPrelimPrecision,
        'SVM Recall': svmPrelimRecall,
        'SVM F1 Score': svmPrelimF1,
        'SVM Mean Squared Error (MSE)': svmPrelimMse,
        'SVM Root Mean Squared Error (RMSE)': svmPrelimRmse
    }
    cors_headers = {
        'Access-Control-Allow-Origin': '*',  # You can set this to a specific origin or origins
        'Access-Control-Allow-Methods': 'POST',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Credentials': 'true',
    }

    return jsonify(response), 200, cors_headers

if __name__ == '__main__':
    app.run(debug=True, port=5000)