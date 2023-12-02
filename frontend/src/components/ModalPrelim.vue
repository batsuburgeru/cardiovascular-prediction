<template>
  <div class="modal fade" id="exampleModalToggle" aria-hidden="true" aria-labelledby="exampleModalToggleLabel" tabindex="-1">
    <div class="modal-dialog modal-dialog-centered">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title" id="exampleModalToggleLabel">Logistic Regression</h5>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body" v-if="data">
          <p>Logistic Regression Predicted Class: {{ data['Logistic Regression Predicted Class'] }}</p>
          <p>Logistic Regression Probability: {{ data['Logistic Regression Probability'] }}</p>
          <p>Logistic Regression Accuracy: {{ data['Logistic Regression Accuracy'] }}</p>
          <p>Logistic Regression Confusion Matrix: {{ data['Logistic Regression Confusion Matrix'] }}</p>
          <p>Logistic Regression Precision: {{ data['Logistic Regression Precision'] }}</p>
          <p>Logistic Regression Recall: {{ data['Logistic Regression Recall'] }}</p>
          <p>Logistic Regression F1 Score: {{ data['Logistic Regression F1 Score'] }}</p>
          <p>Logistic Regression Mean Squared Errod (MSE): {{ data['Logistic Regression Mean Squared Errod (MSE)'] }}</p>
          <p>Logistic Regression Root Mean Squared Error (RMSE): {{ data['Logistic Regression Root Mean Squared Error (RMSE)'] }}</p>
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
        <div class="modal-body">
          Show a third modal and hide this one with the button below.
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
        <div class="modal-body">
          Hide this modal and show the first with the button below.
        </div>
        <div class="modal-footer">
          <button class="btn btn-primary" data-bs-dismiss="modal" @click="navigateToMoreThan">Continue</button>
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
      data: Object
    },
    methods: {
      navigateToMoreThan() {
    this.$router.push('/moreThan');
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
  