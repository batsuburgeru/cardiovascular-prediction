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
        'Logistic_Regression_Predicted_Class': lrPrelimPredictedClass.tolist(),
        'Logistic_Regression_Probability': lrPrelimProbability,
        'Logistic_Regression_Accuracy': lrPrelimAccuracy,
        'Logistic_Regression_Confusion Matrix': lrPrelimConfusionMatrix.tolist(),
        'Logistic_Regression_Precision': lrPrelimPrecision,
        'Logistic_Regression_Recall': lrPrelimRecall,
        'Logistic_Regression_F1_Score': lrPrelimF1,
        'Logistic_Regression_Mean_Squared_Error': lrPrelimMse,
        'Logistic_Regression_Root_Mean_Squared_Error': lrPrelimRmse,

        'KNN_Predicted_Class': knnPrelimPredictedClass.tolist(),
        'KNN_Probability':knnPrelimProbability,
        'KNN_Accuracy': knnPrelimAccuracy,
        'KNN_Confusion_Matrix': knnPrelimConfusionMatrix.tolist(),
        'KNN_Precision': knnPrelimPrecision,
        'KNN_Recall': knnPrelimRecall,
        'KNN_F1_Score': knnPrelimF1,
        'KNN_Mean_Squared_Error': knnPrelimMse,
        'KNN_Root_Mean_Squared_Error': knnPrelimRmse,

        'SVM_Predicted_Class': svmPrelimPredictedClass.tolist(),
        'SVM_Probability':svmPrelimProbability,
        'SVM_Accuracy': svmPrelimAccuracy,
        'SVM_Confusion_Matrix': svmPrelimConfusionMatrix.tolist(),
        'SVM_Precision': svmPrelimPrecision,
        'SVM_Recall': svmPrelimRecall,
        'SVM_F1_Score': svmPrelimF1,
        'SVM_Mean_Squared_Error': svmPrelimMse,
        'SVM_Root_Mean_Squared_Error': svmPrelimRmse
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