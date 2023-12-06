<template>
  <div class="modal fade" id="exampleModalToggle" aria-hidden="true" aria-labelledby="exampleModalToggleLabel" tabindex="-1">
    <div class="modal-dialog modal-dialog-centered">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title" id="exampleModalToggleLabel">Logistic Regression</h5>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body" v-if="responseData">
          <p>Logistic Regression Predicted Class: {{ responseData.Logistic_Regression_Predicted_Class}}</p>
          <p>Logistic Regression Probability: {{ responseData.Logistic_Regression_Probability }}</p>
          <p>Logistic Regression Accuracy: {{ responseData.Logistic_Regression_Accuracy }}</p>
          <p>Logistic Regression Confusion Matrix: {{ responseData.Logistic_Regression_Confusion_Matrix }}</p>
          <p>Logistic Regression Precision: {{ responseData.Logistic_Regression_Precision }}</p>
          <p>Logistic Regression Recall: {{ responseData.Logistic_Regression_Recall }}</p>
          <p>Logistic Regression F1 Score: {{ responseData.Logistic_Regression_F1_Score}}</p>
          <p>Logistic Regression Mean Squared Error (MSE): {{ responseData.Logistic_Regression_Mean_Squared_Error }}</p>
          <p>Logistic Regression Root Mean Squared Error (RMSE): {{ responseData.Logistic_Regression_Root_Mean_Squared_Error }}</p>
        </div>
        <div class="modal-footer">
          <button class="btn btn-primary" data-bs-target="#exampleModalToggle2" data-bs-toggle="modal" data-bs-dismiss="modal">Next</button>
        </div>
      </div>
    </div>
  </div>
  <div class="modal fade" id="exampleModalToggle2" aria-hidden="true" aria-labelledby="exampleModalToggleLabel2" tabindex="-1">
    <div class="modal-dialog modal-dialog-centered">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title" id="exampleModalToggleLabel2">KNN</h5>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body" v-if="responseData">
          <p>KNN Predicted Class: {{ responseData.KNN_Predicted_Class}}</p>
          <p>KNN Probability: {{ responseData.KNN_Probability }}</p>
          <p>KNN Accuracy: {{ responseData.KNN_Accuracy }}</p>
          <p>KNN Confusion Matrix: {{ responseData.KNN_Confusion_Matrix }}</p>
          <p>KNN Precision: {{ responseData.KNN_Precision }}</p>
          <p>KNN Recall: {{ responseData.KNN_Recall }}</p>
          <p>KNN F1 Score: {{ responseData.KNN_F1_Score}}</p>
          <p>KNN Mean Squared Error (MSE): {{ responseData.KNN_Mean_Squared_Error }}</p>
          <p>KNN Root Mean Squared Error (RMSE): {{ responseData.KNN_Root_Mean_Squared_Error }}</p>
        </div>
        <div class="modal-footer">
          <button class="btn btn-primary" data-bs-target="#exampleModalToggle3" data-bs-toggle="modal" data-bs-dismiss="modal">Next</button>
        </div>
      </div>
    </div>
  </div>
  <div class="modal fade" id="exampleModalToggle3" aria-hidden="true" aria-labelledby="exampleModalToggleLabel3" tabindex="-1">
    <div class="modal-dialog modal-dialog-centered">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title" id="exampleModalToggleLabel3">SVM</h5>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body" v-if="responseData">
          <p>SVM Predicted Class: {{ responseData.SVM_Predicted_Class}}</p>
          <p>SVM Probability: {{ responseData.SVM_Probability }}</p>
          <p>SVM Accuracy: {{ responseData.SVM_Accuracy }}</p>
          <p>SVM Confusion Matrix: {{ responseData.SVM_Confusion_Matrix }}</p>
          <p>SVM Precision: {{ responseData.SVM_Precision }}</p>
          <p>SVM Recall: {{ responseData.SVM_Recall }}</p>
          <p>SVM F1 Score: {{ responseData.SVM_F1_Score}}</p>
          <p>SVM Mean Squared Error (MSE): {{ responseData.SVM_Mean_Squared_Error }}</p>
          <p>SVM Root Mean Squared Error (RMSE): {{ responseData.SVM_Root_Mean_Squared_Error }}</p>
        </div>
        <div class="modal-footer">
          <button class="btn btn-primary" data-bs-dismiss="modal" @click="navigateToNextPage">Continue</button>
        </div>
      </div>
    </div>
  </div>
  </template>
    
  <script>
  export default {
    name: 'MyModal',
    props: {
      visible: Boolean,
      responseData: Object
    },
    methods: {
      navigateToNextPage() {
      const currentPath = this.$route.path;
      if (this.responseData.SVM_Probability >= 35) {
        // Check the current URL and navigate accordingly
        if (currentPath === '/moreThan') {
          this.$router.push('/anotherMoreThanRoute');
        } else if (currentPath === '/lessThan') {
          this.$router.push('/anotherLessThanRoute');
        } else {
          this.$router.push('/moreThan');
        }
      } else {
        // Navigate to another route or keep the same logic as needed
        this.$router.push('/lessThan');
      }
  },
      closeModal() {
        this.$emit('update:visible', false);
      },
      openModal() {
        this.modalVisible = true;
      },
    },
  }
  </script>


<style>
.modal-body p {
  display: flex;
  justify-content: space-between;
}
</style>
  