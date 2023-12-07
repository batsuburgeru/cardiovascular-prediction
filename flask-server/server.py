from flask import Flask, request, jsonify, send_from_directory
from cvdModels import knnPreliminary, logisticRegressionPreliminary, svmPreliminary, logisticRegressionMoreThan, knnMoreThan, svmMoreThan, logisticRegressionLessThan, knnLessThan, svmLessThan
from flask_cors import CORS
import numpy as np
from fpdf import FPDF
import os

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
        'Logistic_Regression_Confusion_Matrix': lrPrelimConfusionMatrix.tolist(),
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


@app.route('/moreThan', methods=['POST'])
def moreThan():
    data = request.get_json(force=True)  # get the posted data

    age = data['age']
    gender = data['gender']
    trestbps = data['trestbps']
    history = data['history']
    cp = data['cp']
    chol = data['chol']
    fbs = data['fbs']
    restecg = data['restecg']
    thalach = data['thalach']
    thal = data['thal']

    lrMoreThanPredictedClass, lrMoreThanProbability, lrMoreThanAccuracy, lrMoreThanConfusionMatrix, lrMoreThanPrecision, lrMoreThanRecall, lrMoreThanF1, lrMoreThanMse, lrMoreThanRmse = logisticRegressionMoreThan(age, gender, trestbps, history, cp, chol, fbs, restecg, thalach, thal)
    knnMoreThanPredictedClass, knnMoreThanProbability, knnMoreThanAccuracy, knnMoreThanConfusionMatrix, knnMoreThanPrecision, knnMoreThanRecall, knnMoreThanF1, knnMoreThanMse, knnMoreThanRmse = knnMoreThan(age, gender, trestbps, history, cp, chol, fbs, restecg, thalach, thal)
    svmMoreThanPredictedClass, svmMoreThanProbability, svmMoreThanAccuracy, svmMoreThanConfusionMatrix, svmMoreThanPrecision, svmMoreThanRecall, svmMoreThanF1, svmMoreThanMse, svmMoreThanRmse = svmMoreThan(age, gender, trestbps, history, cp, chol, fbs, restecg, thalach, thal)

    response = {
        'Logistic_Regression_Predicted_Class': lrMoreThanPredictedClass.tolist(),
        'Logistic_Regression_Probability': lrMoreThanProbability,
        'Logistic_Regression_Accuracy': lrMoreThanAccuracy,
        'Logistic_Regression_Confusion_Matrix': lrMoreThanConfusionMatrix.tolist(),
        'Logistic_Regression_Precision': lrMoreThanPrecision,
        'Logistic_Regression_Recall': lrMoreThanRecall,
        'Logistic_Regression_F1_Score': lrMoreThanF1,
        'Logistic_Regression_Mean_Squared_Error': lrMoreThanMse,
        'Logistic_Regression_Root_Mean_Squared_Error': lrMoreThanRmse,

        'KNN_Predicted_Class': knnMoreThanPredictedClass.tolist(),
        'KNN_Probability':knnMoreThanProbability,
        'KNN_Accuracy': knnMoreThanAccuracy,
        'KNN_Confusion_Matrix': knnMoreThanConfusionMatrix.tolist(),
        'KNN_Precision': knnMoreThanPrecision,
        'KNN_Recall': knnMoreThanRecall,
        'KNN_F1_Score': knnMoreThanF1,
        'KNN_Mean_Squared_Error': knnMoreThanMse,
        'KNN_Root_Mean_Squared_Error': knnMoreThanRmse,

        'SVM_Predicted_Class': svmMoreThanPredictedClass.tolist(),
        'SVM_Probability':svmMoreThanProbability,
        'SVM_Accuracy': svmMoreThanAccuracy,
        'SVM_Confusion_Matrix': svmMoreThanConfusionMatrix.tolist(),
        'SVM_Precision': svmMoreThanPrecision,
        'SVM_Recall': svmMoreThanRecall,
        'SVM_F1_Score': svmMoreThanF1,
        'SVM_Mean_Squared_Error': svmMoreThanMse,
        'SVM_Root_Mean_Squared_Error': svmMoreThanRmse
    }
    cors_headers = {
        'Access-Control-Allow-Origin': '*',  # You can set this to a specific origin or origins
        'Access-Control-Allow-Methods': 'POST',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Credentials': 'true',
    }

    return jsonify(response), 200, cors_headers

@app.route('/lessThan', methods=['POST'])
def lessThan():
    data = request.get_json(force=True)  # get the posted data

    age = data['age']
    gender = data['gender']
    trestbps = data['trestbps']
    history = data['history']
    cp = data['cp']
    chol = data['chol']
    fbs = data['fbs']
    restecg = data['restecg']

    lrLessThanPredictedClass, lrLessThanProbability, lrLessThanAccuracy, lrLessThanConfusionMatrix, lrLessThanPrecision, lrLessThanRecall, lrLessThanF1, lrLessThanMse, lrLessThanRmse = logisticRegressionLessThan(age, gender, trestbps, history, cp, chol, fbs, restecg)
    knnLessThanPredictedClass, knnLessThanProbability, knnLessThanAccuracy, knnLessThanConfusionMatrix, knnLessThanPrecision, knnLessThanRecall, knnLessThanF1, knnLessThanMse, knnLessThanRmse = knnLessThan(age, gender, trestbps, history, cp, chol, fbs, restecg)
    svmLessThanPredictedClass, svmLessThanProbability, svmLessThanAccuracy, svmLessThanConfusionMatrix, svmLessThanPrecision, svmLessThanRecall, svmLessThanF1, svmLessThanMse, svmLessThanRmse = svmLessThan(age, gender, trestbps, history, cp, chol, fbs, restecg)

    response = {
        'Logistic_Regression_Predicted_Class': lrLessThanPredictedClass.tolist(),
        'Logistic_Regression_Probability': lrLessThanProbability,
        'Logistic_Regression_Accuracy': lrLessThanAccuracy,
        'Logistic_Regression_Confusion_Matrix': lrLessThanConfusionMatrix.tolist(),
        'Logistic_Regression_Precision': lrLessThanPrecision,
        'Logistic_Regression_Recall': lrLessThanRecall,
        'Logistic_Regression_F1_Score': lrLessThanF1,
        'Logistic_Regression_Mean_Squared_Error': lrLessThanMse,
        'Logistic_Regression_Root_Mean_Squared_Error': lrLessThanRmse,

        'KNN_Predicted_Class': knnLessThanPredictedClass.tolist(),
        'KNN_Probability':knnLessThanProbability,
        'KNN_Accuracy': knnLessThanAccuracy,
        'KNN_Confusion_Matrix': knnLessThanConfusionMatrix.tolist(),
        'KNN_Precision': knnLessThanPrecision,
        'KNN_Recall': knnLessThanRecall,
        'KNN_F1_Score': knnLessThanF1,
        'KNN_Mean_Squared_Error': knnLessThanMse,
        'KNN_Root_Mean_Squared_Error': knnLessThanRmse,

        'SVM_Predicted_Class': svmLessThanPredictedClass.tolist(),
        'SVM_Probability':svmLessThanProbability,
        'SVM_Accuracy': svmLessThanAccuracy,
        'SVM_Confusion_Matrix': svmLessThanConfusionMatrix.tolist(),
        'SVM_Precision': svmLessThanPrecision,
        'SVM_Recall': svmLessThanRecall,
        'SVM_F1_Score': svmLessThanF1,
        'SVM_Mean_Squared_Error': svmLessThanMse,
        'SVM_Root_Mean_Squared_Error': svmLessThanRmse
    }
    cors_headers = {
        'Access-Control-Allow-Origin': '*',  # You can set this to a specific origin or origins
        'Access-Control-Allow-Methods': 'POST',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Credentials': 'true',
    }

    return jsonify(response), 200, cors_headers

@app.route('/getImages', methods=['GET'])
def get_images():
    try:
        image_names = os.listdir('src/assets/visualizations')
        image_urls = [f'http://localhost:5000/sendImage/{image}' for image in image_names]
        return jsonify({'image_urls': image_urls})
    except Exception as e:
        return str(e)

@app.route('/sendImage/<image_name>', methods=['GET'])
def send_image(image_name):
    try:
        return send_from_directory('src/assets/visualizations', image_name)
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(debug=True, port=5000)