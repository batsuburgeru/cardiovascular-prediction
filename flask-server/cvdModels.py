from itertools import combinations
from matplotlib import colormaps
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

#KNN PRELIMINARY

def knnPreliminary(age, gender, trestbps, history, cp):
    # Read the dataset
    datasetKnnPrelim = pd.read_csv('./src/assets/heart_attack.csv')
    datasetKnnPrelim['history'] = datasetKnnPrelim['heart_disease'].copy()

    # Features and labels
    X = datasetKnnPrelim[['age', 'gender', 'trestbps', 'history', 'cp']]
    y = datasetKnnPrelim['heart_disease']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the KNN model
    knn = KNeighborsClassifier(n_neighbors=12)
    knn.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Create a new sample for prediction with feature names
    userInput = pd.DataFrame([[age, gender, trestbps, history, cp]],
                             columns=['age', 'gender', 'trestbps', 'history', 'cp'])

    # Make a prediction on the new sample
    prediction = knn.predict(userInput)
    knnPrelimPredictedClass = prediction[0]
    print(f"Predicted Class: {knnPrelimPredictedClass}")

    # Get the probability of heart attack
    probabilities = knn.predict_proba(userInput)
    knnPrelimProbability = round(probabilities[0, 1] * 100, 2)
    print(f"Probability of Heart Attack: {knnPrelimProbability}%")

    # Evaluate the model's accuracy on the test set
    knnPrelimAccuracy = round(knn.score(X_test, y_test)*100,2)
    print(f"Accuracy: {knnPrelimAccuracy}%")

    # Display the confusion matrix
    knnPrelimConfusionMatrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(knnPrelimConfusionMatrix)
    
    knnPrelimPrecision = round(precision_score(y_test, y_pred)*100,2)
    print(f"Precision: {knnPrelimPrecision}%")
    
    knnPrelimRecall = round(recall_score(y_test, y_pred)*100,2)
    print(f"Recall: {knnPrelimRecall}%")
    
    knnPrelimF1 = round(f1_score(y_test, y_pred)*100,2)
    print(f"F1 Score: {knnPrelimF1}%")

    knnPrelimMse = round(mean_squared_error(y_test, y_pred),5)
    print(f"Mean Squared Error (MSE): {knnPrelimMse}")
    
    knnPrelimRmse = round(np.sqrt(knnPrelimMse),5)
    print(f"Root Mean Squared Error (RMSE): {knnPrelimRmse}")
    
    return knnPrelimPredictedClass, knnPrelimProbability, knnPrelimAccuracy, knnPrelimConfusionMatrix, knnPrelimPrecision, knnPrelimRecall, knnPrelimF1, knnPrelimMse, knnPrelimRmse

#KNN MORE THAN 35%

def knnMoreThan(age, gender, restbps, history, cp, chol, fbs, restecg, thalach, thal):
    # Read the dataset
    datasetKnnMoreThan = pd.read_csv('./src/assets/heart_attack.csv')
    datasetKnnMoreThan['history'] = datasetKnnMoreThan['heart_disease'].copy()

    # Features and labels
    X = datasetKnnMoreThan[['age', 'gender', 'trestbps', 'history', 'cp', 'chol', 'fbs', 'restecg', 'thalach', 'thal']]
    y = datasetKnnMoreThan['heart_disease']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the KNN model
    knn = KNeighborsClassifier(n_neighbors=12)
    knn.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Create a new sample for prediction with feature names
    userInput = pd.DataFrame([[age, gender, restbps, history, cp, chol, fbs, restecg, thalach, thal]],
                             columns=['age', 'gender', 'trestbps', 'history', 'cp', 'chol', 'fbs', 'restecg', 'thalach', 'thal'])

    # Make a prediction on the new sample
    prediction = knn.predict(userInput)
    knnMoreThanPredictedClass = prediction[0]
    print(f"Predicted Class: {knnMoreThanPredictedClass}")

    # Get the probability of heart attack
    probabilities = knn.predict_proba(userInput)
    knnMoreThanProbability = round(probabilities[0, 1] * 100, 2)
    print(f"Probability of Heart Attack: {knnMoreThanProbability}%")

    # Evaluate the model's accuracy on the test set
    knnMoreThanAccuracy = round(knn.score(X_test, y_test)*100,2)
    print(f"Accuracy: {knnMoreThanAccuracy}%")

    # Display the confusion matrix
    knnMoreThanConfusionMatrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(knnMoreThanConfusionMatrix)
    
    knnMoreThanPrecision = round(precision_score(y_test, y_pred)*100,2)
    print(f"Precision: {knnMoreThanPrecision}%")
    
    knnMoreThanRecall = round(recall_score(y_test, y_pred)*100,2)
    print(f"Recall: {knnMoreThanRecall}%")
    
    knnMoreThanF1 = round(f1_score(y_test, y_pred)*100,2)
    print(f"F1 Score: {knnMoreThanF1}%")

    knnMoreThanMse = round(mean_squared_error(y_test, y_pred),5)
    print(f"Mean Squared Error (MSE): {knnMoreThanMse}")
    
    knnMoreThanRmse = round(np.sqrt(knnMoreThanMse),5)
    print(f"Root Mean Squared Error (RMSE): {knnMoreThanRmse}")
    
    def plot_scatter_matrix(data, target):
        # Add the target variable to the dataset
        data_with_target = data.copy()
        data_with_target['heart_disease'] = target

        # Select a subset of features for visualization
        selected_features = X

        # Create a pairplot for the selected features
        sns.pairplot(data_with_target, vars=selected_features, hue='heart_disease', palette='husl', markers=['o', 's'])
        plt.suptitle('Scatterplot Matrix for Selected Features', y=1.02)

        plt.savefig("src/assets/visualizations/knnResult.png")
        
    plot_scatter_matrix(X, y)   
    
    return knnMoreThanPredictedClass, knnMoreThanProbability, knnMoreThanAccuracy, knnMoreThanConfusionMatrix, knnMoreThanPrecision, knnMoreThanRecall, knnMoreThanF1, knnMoreThanMse, knnMoreThanRmse

#KNN LESS THAN 35%

def knnLessThan(age, gender, restbps, history, cp, chol, fbs, restecg):
    datasetKnnLessThan = pd.read_csv('./src/assets/heart_attack.csv')
    datasetKnnLessThan['history'] = datasetKnnLessThan['heart_disease'].copy()

    # Features and labels
    X = datasetKnnLessThan[['age', 'gender', 'trestbps', 'history', 'cp', 'chol', 'fbs', 'restecg']]
    y = datasetKnnLessThan['heart_disease']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the KNN model
    knn = KNeighborsClassifier(n_neighbors=12)
    knn.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Create a new sample for prediction with feature names
    userInput = pd.DataFrame([[age, gender, restbps, history, cp, chol, fbs, restecg]],
                             columns=['age', 'gender', 'trestbps', 'history', 'cp', 'chol', 'fbs', 'restecg'])

    # Make a prediction on the new sample
    prediction = knn.predict(userInput)
    knnLessThanPredictedClass = prediction[0]
    print(f"Predicted Class: {knnLessThanPredictedClass}")

    # Get the probability of heart attack
    probabilities = knn.predict_proba(userInput)
    knnLessThanProbability = round(probabilities[0, 1] * 100, 2)
    print(f"Probability of Heart Attack: {knnLessThanProbability}%")

    # Evaluate the model's accuracy on the test set
    knnLessThanAccuracy = round(knn.score(X_test, y_test)*100,2)
    print(f"Accuracy: {knnLessThanAccuracy}%")

    # Display the confusion matrix
    knnLessThanConfusionMatrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(knnLessThanConfusionMatrix)
    
    knnLessThanPrecision = round(precision_score(y_test, y_pred)*100,2)
    print(f"Precision: {knnLessThanPrecision}%")
    
    knnLessThanRecall = round(recall_score(y_test, y_pred)*100,2)
    print(f"Recall: {knnLessThanRecall}%")
    
    knnLessThanF1 = round(f1_score(y_test, y_pred)*100,2)
    print(f"F1 Score: {knnLessThanF1}%")

    knnLessThanMse = round(mean_squared_error(y_test, y_pred),5)
    print(f"Mean Squared Error (MSE): {knnLessThanMse}")
    
    knnLessThanRmse = round(np.sqrt(knnLessThanMse),5)
    print(f"Root Mean Squared Error (RMSE): {knnLessThanRmse}")
    
    def plot_scatter_matrix(data, target):
        # Add the target variable to the dataset
        data_with_target = data.copy()
        data_with_target['heart_disease'] = target

        # Select a subset of features for visualization
        selected_features = X

        # Create a pairplot for the selected features
        sns.pairplot(data_with_target, vars=selected_features, hue='heart_disease', palette='husl', markers=['o', 's'])
        plt.suptitle('Scatterplot Matrix for Selected Features', y=1.02)
        
        plt.savefig("src/assets/visualizations/knnResult.png")
        
    plot_scatter_matrix(X, y)
    
    return knnLessThanPredictedClass, knnLessThanProbability, knnLessThanAccuracy, knnLessThanConfusionMatrix, knnLessThanPrecision, knnLessThanRecall, knnLessThanF1, knnLessThanMse, knnLessThanRmse

#SVM PRELIMINARY

def svmPreliminary(age, gender, trestbps, history, cp):
    datasetSvmPrelim = pd.read_csv('./src/assets/heart_attack.csv')
    datasetSvmPrelim['history'] = datasetSvmPrelim['heart_disease'].copy()

    X = datasetSvmPrelim[['age', 'gender', 'trestbps', 'history', 'cp']]
    y = datasetSvmPrelim['heart_disease']

    scaler = StandardScaler()
    
    # Use a DataFrame for X to preserve column names
    X_df = pd.DataFrame(X, columns=['age', 'gender', 'trestbps', 'history', 'cp'])
    X_standardized = scaler.fit_transform(X_df)

    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

    svm = SVC(kernel='linear', probability = True)

    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)

    userInput = pd.DataFrame([[age, gender, trestbps, history, cp]],
                             columns=['age', 'gender', 'trestbps', 'history', 'cp'])
    
    userInputStandardized = scaler.transform(userInput)

    
    prediction = svm.predict(userInputStandardized)
    svmPrelimPredictedClass = prediction[0]
    print(f"Predicted Class: {svmPrelimPredictedClass}")
    
    probabilities = svm.predict_proba(userInputStandardized)
    svmPrelimProbability = round(probabilities[0, 1] * 100, 2)
    print(f"Probability of Heart Disease: {svmPrelimProbability}%")
    
    # Evaluate the model's accuracy on the test set
    svmPrelimAccuracy = round(svm.score(X_test, y_test)*100,2)
    print(f"Accuracy: {svmPrelimAccuracy}%")

    # Display the confusion matrix
    svmPrelimConfusionMatrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(svmPrelimConfusionMatrix)
    
    svmPrelimPrecision = round(precision_score(y_test, y_pred)*100,2)
    print(f"Precision: {svmPrelimPrecision}%")
    
    svmPrelimRecall = round(recall_score(y_test, y_pred)*100,2)
    print(f"Recall: {svmPrelimRecall}%")
    
    svmPrelimF1 = round(f1_score(y_test, y_pred)*100,2)
    print(f"F1 Score: {svmPrelimF1}%")

    svmPrelimMse = round(mean_squared_error(y_test, y_pred),5)
    print(f"Mean Squared Error (MSE): {svmPrelimMse}")
    
    svmPrelimRmse = round(np.sqrt(svmPrelimMse),5)
    print(f"Root Mean Squared Error (RMSE): {svmPrelimRmse}")
    
    return svmPrelimPredictedClass, svmPrelimProbability, svmPrelimAccuracy, svmPrelimConfusionMatrix, svmPrelimPrecision, svmPrelimRecall, svmPrelimF1, svmPrelimMse, svmPrelimRmse

#SVM More Than 35%

def svmMoreThan(age, gender, restbps, history, cp, chol, fbs, restecg, thalach, thal):
    datasetSvmMoreThan = pd.read_csv('./src/assets/heart_attack.csv')
    datasetSvmMoreThan['history'] = datasetSvmMoreThan['heart_disease'].copy()

    X = datasetSvmMoreThan[['age', 'gender', 'trestbps', 'history', 'cp', 'chol', 'fbs', 'restecg', 'thalach', 'thal']]
    y = datasetSvmMoreThan['heart_disease']

    scaler = StandardScaler()
    
    # Use a DataFrame for X to preserve column names
    X_df = pd.DataFrame(X, columns=['age', 'gender', 'trestbps', 'history', 'cp', 'chol', 'fbs', 'restecg', 'thalach', 'thal'])
    X_standardized = scaler.fit_transform(X_df)

    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

    svm = SVC(kernel='linear', C = 1.0, probability = True)

    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)

    userInput = pd.DataFrame([[age, gender, restbps, history, cp, chol, fbs, restecg, thalach, thal]],
                             columns=['age', 'gender', 'trestbps', 'history', 'cp', 'chol', 'fbs', 'restecg', 'thalach', 'thal'])
    userInputStandardized = scaler.transform(userInput)
    
    prediction = svm.predict(userInputStandardized)
    svmMoreThanPredictedClass = prediction[0]
    print(f"Predicted Class: {svmMoreThanPredictedClass}")
    
    probabilities = svm.predict_proba(userInputStandardized)
    svmMoreThanProbability = round(probabilities[0,1] * 100, 2)
    print(f"Probability of Heart Disease: {svmMoreThanProbability}%")
    
    # Evaluate the model's accuracy on the test set
    svmMoreThanAccuracy = round(svm.score(X_test, y_test)*100,2)
    print(f"Accuracy: {svmMoreThanAccuracy}%")

    # Display the confusion matrix
    svmMoreThanConfusionMatrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(svmMoreThanConfusionMatrix)
    
    svmMoreThanPrecision = round(precision_score(y_test, y_pred)*100,2)
    print(f"Precision: {svmMoreThanPrecision}%")
    
    svmMoreThanRecall = round(recall_score(y_test, y_pred)*100,2)
    print(f"Recall: {svmMoreThanRecall}%")
    
    svmMoreThanF1 = round(f1_score(y_test, y_pred)*100,2)
    print(f"F1 Score: {svmMoreThanF1}%")

    svmMoreThanMse = round(mean_squared_error(y_test, y_pred),5)
    print(f"Mean Squared Error (MSE): {svmMoreThanMse}")
    
    svmMoreThanRmse = round(np.sqrt(svmMoreThanMse),5)
    print(f"Root Mean Squared Error (RMSE): {svmMoreThanRmse}")
    
    X_std = scaler.fit_transform(X)
    feature_combinations = list(combinations(range(X.shape[1]),2))
    
    feature_names = X_df.columns
    
    num_subplots = len(feature_combinations)
    num_rows = int(np.ceil(num_subplots / 3))
    
    fig, axs = plt.subplots(num_rows, 3, figsize=(20, 4 * num_rows))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    axs = axs.flatten()
    
    for i, (feature_idx1, feature_idx2) in enumerate(feature_combinations):
        X_pair_std = X_std[:, [feature_idx1, feature_idx2]]
        
        svm.fit(X_pair_std, y)
        
        x_min, x_max = X_pair_std[:, 0].min() - 1, X_pair_std[:, 0].max() + 1
        y_min, y_max = X_pair_std[:, 1].min() - 1, X_pair_std[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
        
        Z = svm.predict(np.array([xx.ravel(), yy.ravel()]).T)
        Z = Z.reshape(xx.shape)
        
        axs[i].pcolormesh(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
        axs[i].scatter(X_pair_std[:, 0], X_pair_std[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
        axs[i].set_xlabel(feature_names[feature_idx1], fontsize=14)
        axs[i].set_ylabel(feature_names[feature_idx2], fontsize=14)
        axs[i].tick_params(axis='both', which='major', labelsize=12)
        axs[i].set_title('SVM Decision Boundary', fontsize=16)
        
    for j in range(num_subplots, len(axs)):
        axs[j].axis('off')
    
    plt.savefig("src/assets/visualizations/svmResult.png")
        
    return svmMoreThanPredictedClass, svmMoreThanProbability, svmMoreThanAccuracy, svmMoreThanConfusionMatrix, svmMoreThanPrecision, svmMoreThanRecall, svmMoreThanF1, svmMoreThanMse, svmMoreThanRmse

#SVM LESS THAN 35%

def svmLessThan(age, gender, restbps, history, cp, chol, fbs, restecg):
    datasetSvmLessThan = pd.read_csv('./src/assets/heart_attack.csv')
    datasetSvmLessThan['history'] = datasetSvmLessThan['heart_disease'].copy()

    X = datasetSvmLessThan[['age', 'gender', 'trestbps', 'history', 'cp', 'chol', 'fbs', 'restecg']]
    y = datasetSvmLessThan['heart_disease']

    scaler = StandardScaler()
    
    # Use a DataFrame for X to preserve column names
    X_df = pd.DataFrame(X, columns=['age', 'gender', 'trestbps', 'history', 'cp', 'chol', 'fbs', 'restecg'])
    X_standardized = scaler.fit_transform(X_df)

    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

    svm = SVC(kernel='linear', probability = True)

    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)

    userInput = pd.DataFrame([[age, gender, restbps, history, cp, chol, fbs, restecg]],
                             columns=['age', 'gender', 'trestbps', 'history', 'cp', 'chol', 'fbs', 'restecg'])
    userInputStandardized = scaler.transform(userInput)

    
    prediction = svm.predict(userInputStandardized)
    svmLessThanPredictedClass = prediction[0]
    print(f"Predicted Class: {svmLessThanPredictedClass}")
    
    probabilities = svm.predict_proba(userInputStandardized)
    svmLessThanProbability = round(probabilities[0, 1] * 100, 2)
    print(f"Probability of Heart Disease: {svmLessThanProbability}%")
    
    # Evaluate the model's accuracy on the test set
    svmLessThanAccuracy = round(svm.score(X_test, y_test)*100,2)
    print(f"Accuracy: {svmLessThanAccuracy}%")

    # Display the confusion matrix
    svmLessThanConfusionMatrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(svmLessThanConfusionMatrix)
    
    svmLessThanPrecision = round(precision_score(y_test, y_pred)*100,2)
    print(f"Precision: {svmLessThanPrecision}%")
    
    svmLessThanRecall = round(recall_score(y_test, y_pred)*100,2)
    print(f"Recall: {svmLessThanRecall}%")
    
    svmLessThanF1 = round(f1_score(y_test, y_pred)*100,2)
    print(f"F1 Score: {svmLessThanF1}%")

    svmLessThanMse = round(mean_squared_error(y_test, y_pred),5)
    print(f"Mean Squared Error (MSE): {svmLessThanMse}")
    
    svmLessThanRmse = round(np.sqrt(svmLessThanMse),5)
    print(f"Root Mean Squared Error (RMSE): {svmLessThanRmse}")
    
    X_std = scaler.fit_transform(X)
    feature_combinations = list(combinations(range(X.shape[1]),2))
    
    feature_names = X_df.columns
    
    num_subplots = len(feature_combinations)
    num_rows = int(np.ceil(num_subplots / 3))
    
    fig, axs = plt.subplots(num_rows, 3, figsize=(20, 4 * num_rows))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    axs = axs.flatten()
    
    for i, (feature_idx1, feature_idx2) in enumerate(feature_combinations):
        X_pair_std = X_std[:, [feature_idx1, feature_idx2]]
        
        svm.fit(X_pair_std, y)
        
        x_min, x_max = X_pair_std[:, 0].min() - 1, X_pair_std[:, 0].max() + 1
        y_min, y_max = X_pair_std[:, 1].min() - 1, X_pair_std[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
        
        Z = svm.predict(np.array([xx.ravel(), yy.ravel()]).T)
        Z = Z.reshape(xx.shape)
        
        axs[i].pcolormesh(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
        axs[i].scatter(X_pair_std[:, 0], X_pair_std[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
        axs[i].set_xlabel(feature_names[feature_idx1], fontsize=14)
        axs[i].set_ylabel(feature_names[feature_idx2], fontsize=14)
        axs[i].tick_params(axis='both', which='major', labelsize=12)
        axs[i].set_title('SVM Decision Boundary', fontsize=16)
        
    for j in range(num_subplots, len(axs)):
        axs[j].axis('off')
    
    plt.savefig("src/assets/visualizations/svmResult.png")
    
    return svmLessThanPredictedClass, svmLessThanProbability, svmLessThanAccuracy, svmLessThanConfusionMatrix, svmLessThanPrecision, svmLessThanRecall, svmLessThanF1, svmLessThanMse, svmLessThanRmse

#LOGISTIC REGRESSION PRELIMINARY

def logisticRegressionPreliminary(age, gender, restbps, history, cp):
    datasetLogisticRegressionPrelim = pd.read_csv('./src/assets/heart_attack.csv')
    datasetLogisticRegressionPrelim['history'] = datasetLogisticRegressionPrelim['heart_disease'].copy()

    X = datasetLogisticRegressionPrelim[['age', 'gender', 'trestbps', 'history', 'cp']]
    y = datasetLogisticRegressionPrelim['heart_disease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    # Make predictions for the given inputs
    userInput = pd.DataFrame([[age, gender, restbps, history, cp]],
                             columns=['age', 'gender', 'trestbps', 'history', 'cp'])

    userInputStandardized = scaler.transform(userInput)

    prediction = classifier.predict(userInputStandardized)
    lrPrelimPredictedClass = prediction[0]
    print(f"Predicted Class: {lrPrelimPredictedClass}")

    probabilities = classifier.predict_proba(userInputStandardized)
    lrPrelimProbability = round(probabilities[0, 1] * 100, 2)
    print(f"Probability of Heart Disease: {lrPrelimProbability}%")
    
    # Evaluate the model's accuracy on the test set
    lrPrelimAccuracy = round(classifier.score(X_test, y_test)*100,2)
    print(f"Accuracy: {lrPrelimAccuracy}%")

    # Display the confusion matrix
    lrPrelimConfusionMatrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(lrPrelimConfusionMatrix)
    
    lrPrelimPrecision = round(precision_score(y_test, y_pred)*100,2)
    print(f"Precision: {lrPrelimPrecision}%")
    
    lrPrelimRecall = round(recall_score(y_test, y_pred)*100,2)
    print(f"Recall: {lrPrelimRecall}%")
    
    lrPrelimF1 = round(f1_score(y_test, y_pred)*100,2)
    print(f"F1 Score: {lrPrelimF1}%")

    lrPrelimMse = round(mean_squared_error(y_test, y_pred),5)
    print(f"Mean Squared Error (MSE): {lrPrelimMse}")
    
    lrPrelimRmse = round(np.sqrt(lrPrelimMse),5)
    print(f"Root Mean Squared Error (RMSE): {lrPrelimRmse}")
    
    return lrPrelimPredictedClass, lrPrelimProbability, lrPrelimAccuracy, lrPrelimConfusionMatrix, lrPrelimPrecision, lrPrelimRecall, lrPrelimF1, lrPrelimMse, lrPrelimRmse

#LOGISTIC REGRESSION MORE THAN 35%

def logisticRegressionMoreThan(age, gender, restbps, history, cp, chol, fbs, restecg, thalach, thal):
    datasetLrMoreThan = pd.read_csv('./src/assets/heart_attack.csv')
    datasetLrMoreThan['history'] = datasetLrMoreThan['heart_disease'].copy()

    X = datasetLrMoreThan[['age', 'gender', 'trestbps', 'history', 'cp', 'chol', 'fbs', 'restecg', 'thalach', 'thal']]
    y = datasetLrMoreThan['heart_disease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    # Make predictions for the given inputs
    userInput = pd.DataFrame([[age, gender, restbps, history, cp, chol, fbs, restecg, thalach, thal]],
                             columns=['age', 'gender', 'trestbps', 'history', 'cp', 'chol', 'fbs', 'restecg', 'thalach', 'thal'])

    userInputStandardized = scaler.transform(userInput)

    prediction = classifier.predict(userInputStandardized)
    lrMoreThanPredictedClass = prediction[0]
    print(f"Predicted Class: {lrMoreThanPredictedClass}")

    probabilities = classifier.predict_proba(userInputStandardized)
    lrMoreThanProbability = round(probabilities[0, 1] * 100, 2)
    print(f"Probability of Heart Disease: {lrMoreThanProbability}%")
    
    # Evaluate the model's accuracy on the test set
    lrMoreThanAccuracy = round(classifier.score(X_test, y_test)*100,2)
    print(f"Accuracy: {lrMoreThanAccuracy}%")

    # Display the confusion matrix
    lrMoreThanConfusionMatrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(lrMoreThanConfusionMatrix)
    
    lrMoreThanPrecision = round(precision_score(y_test, y_pred)*100,2)
    print(f"Precision: {lrMoreThanPrecision}%")
    
    lrMoreThanRecall = round(recall_score(y_test, y_pred)*100,2)
    print(f"Recall: {lrMoreThanRecall}%")
    
    lrMoreThanF1 = round(f1_score(y_test, y_pred)*100,2)
    print(f"F1 Score: {lrMoreThanF1}%")

    lrMoreThanMse = round(mean_squared_error(y_test, y_pred),5)
    print(f"Mean Squared Error (MSE): {lrMoreThanMse}")
    
    lrMoreThanRmse = round(np.sqrt(lrMoreThanMse),5)
    print(f"Root Mean Squared Error (RMSE): {lrMoreThanRmse}")
    
    def plot_scatter_matrix(data, target):
        # Add the target variable to the dataset
        data_with_target = data.copy()
        data_with_target['heart_disease'] = target

        # Select a subset of features for visualization
        selected_features = X

        # Create a pairplot for the selected features
        sns.pairplot(data_with_target, vars=selected_features, hue='heart_disease', palette='husl', markers=['o', 's'])
        plt.suptitle('Scatterplot Matrix for Selected Features', y=1.02)
        
        plt.savefig("src/assets/visualizations/logRegResult.png")
        
    plot_scatter_matrix(X, y)
    
    return lrMoreThanPredictedClass, lrMoreThanProbability, lrMoreThanAccuracy, lrMoreThanConfusionMatrix, lrMoreThanPrecision, lrMoreThanRecall, lrMoreThanF1, lrMoreThanMse, lrMoreThanRmse

#LOGISTIC REGRESSION LESS THAN 35%

def logisticRegressionLessThan(age, gender, restbps, history, cp, chol, fbs, restecg):
    datasetLrLessThan = pd.read_csv('./src/assets/heart_attack.csv')
    datasetLrLessThan['history'] = datasetLrLessThan['heart_disease'].copy()

    X = datasetLrLessThan[['age', 'gender', 'trestbps', 'history', 'cp', 'chol', 'fbs', 'restecg']]
    y = datasetLrLessThan['heart_disease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    # Make predictions for the given inputs
    userInput = pd.DataFrame([[age, gender, restbps, history, cp, chol, fbs, restecg]],
                             columns=['age', 'gender', 'trestbps', 'history', 'cp', 'chol', 'fbs', 'restecg'])

    userInputStandardized = scaler.transform(userInput)

    prediction = classifier.predict(userInputStandardized)
    lrLessThanPredictedClass = prediction[0]
    print(f"Predicted Class: {lrLessThanPredictedClass}")

    probabilities = classifier.predict_proba(userInputStandardized)
    lrLessThanProbability = round(probabilities[0, 1] * 100, 2)
    print(f"Probability of Heart Disease: {lrLessThanProbability}%")
    
    # Evaluate the model's accuracy on the test set
    lrLessThanAccuracy = round(classifier.score(X_test, y_test)*100,2)
    print(f"Accuracy: {lrLessThanAccuracy}%")

    # Display the confusion matrix
    lrLessThanConfusionMatrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(lrLessThanConfusionMatrix)
    
    lrLessThanPrecision = round(precision_score(y_test, y_pred)*100,2)
    print(f"Precision: {lrLessThanPrecision}%")
    
    lrLessThanRecall = round(recall_score(y_test, y_pred)*100,2)
    print(f"Recall: {lrLessThanRecall}%")
    
    lrLessThanF1 = round(f1_score(y_test, y_pred)*100,2)
    print(f"F1 Score: {lrLessThanF1}%")

    lrLessThanMse = round(mean_squared_error(y_test, y_pred),5)
    print(f"Mean Squared Error (MSE): {lrLessThanMse}")
    
    lrLessThanRmse = round(np.sqrt(lrLessThanMse),5)
    print(f"Root Mean Squared Error (RMSE): {lrLessThanRmse}")
    
    def plot_scatter_matrix(data, target):
        # Add the target variable to the dataset
        data_with_target = data.copy()
        data_with_target['heart_disease'] = target

        # Select a subset of features for visualization
        selected_features = X

        # Create a pairplot for the selected features
        sns.pairplot(data_with_target, vars=selected_features, hue='heart_disease', palette='husl', markers=['o', 's'])
        plt.suptitle('Scatterplot Matrix for Selected Features', y=1.02)
        
        plt.savefig("src/assets/visualizations/logRegResult.png")
        
    plot_scatter_matrix(X, y)
    
    return lrLessThanPredictedClass, lrLessThanProbability, lrLessThanAccuracy, lrLessThanConfusionMatrix, lrLessThanPrecision, lrLessThanRecall, lrLessThanF1, lrLessThanMse, lrLessThanRmse

