<template>
  <main>
    <NavBar class="no-print"/>
    <div class="row align-items-start mx-2">
      <h2 class="mb-5 fw-bold"> EVALUATION REPORT </h2>
    <div class="col">
      <div class="mx-auto" style="width: 75%;">
        <h3 class="fw-bold mb-5">INPUT DATA</h3>
        <div v-if="finalData" class="row text-start">
          <dl class="row">
            <dt class="col-sm-10 dt-no-wrap">Age:</dt>
            <dd class="col-sm-2">{{ finalData.age }}</dd>

            <dt class="col-sm-10 dt-no-wrap">Gender:</dt>
            <dd class="col-sm-2">{{ finalData.gender }}</dd>

            <dt class="col-sm-10">Resting Blood Pressure:</dt>
            <dd class="col-sm-2">{{ finalData.trestbps }}</dd>

            <dt class="col-sm-10">History of Heart Disease:</dt>
            <dd class="col-sm-2">{{ finalData.history }}</dd>

            <dt class="col-sm-10 dt-no-wrap">Chest Pain Type:</dt>
            <dd class="col-sm-2">{{ finalData.cp }}</dd>

            <dt class="col-sm-10 dt-no-wrap">Cholesterol:</dt>
            <dd class="col-sm-2">{{ finalData.chol }}</dd>

            <dt class="col-sm-10 dt-no-wrap">Fasting Blood Sugar:</dt>
            <dd class="col-sm-2">{{ finalData.fbs }}</dd>

            <dt class="col-sm-10">Resting Cardiographic Result:</dt>
            <dd class="col-sm-2">{{ finalData.restecg }}</dd>

            <dt class="col-sm-10" v-if="finalData.thalach">Maximum Heart Rate Achieved:</dt>
            <dd class="col-sm-2" v-if="finalData.thalach">{{ finalData.thalach }}</dd>

            <dt class="col-sm-10 dt-no-wrap" v-if="finalData.thal">Thalassemia Type:</dt>
            <dd class="col-sm-2" v-if="finalData.thal">{{ finalData.thal }}</dd>
          </dl>
        </div>
      </div>
    </div>
    <div class="col">
      <div class="mx-auto" style="width: 100%;">
        <h3 class="fw-bold mb-5">SVM RESULT</h3>
        <div v-if="responseData" class="row text-start">
          <dl class="row">
            <dt class="col-sm-8 dt-no-wrap">SVM Predicted Class:</dt>
            <dd class="col-sm-2">{{ responseData.SVM_Predicted_Class }}</dd>

            <dt class="col-sm-8 dt-no-wrap">SVM Probability:</dt>
            <dd class="col-sm-2">{{ responseData.SVM_Probability }}%</dd>

            <dt class="col-sm-8 dt-no-wrap">SVM Accuracy:</dt>
            <dd class="col-sm-2">{{ responseData.SVM_Accuracy }}%</dd>

            <dt class="col-sm-8 dt-no-wrap">SVM Confusion Matrix:</dt>
            <dd class="col-sm-2 dt-no-wrap">{{ responseData.SVM_Confusion_Matrix[0] }}</dd>
            <dt class="col-sm-8 dt-no-wrap"></dt>
            <dd class="col-sm-2 dt-no-wrap">{{ responseData.SVM_Confusion_Matrix[1] }}</dd>

            <dt class="col-sm-8 dt-no-wrap">SVM Precision:</dt>
            <dd class="col-sm-2">{{ responseData.SVM_Precision}}%</dd>

            <dt class="col-sm-8 dt-no-wrap">SVM Recall:</dt>
            <dd class="col-sm-2">{{ responseData.SVM_Recall }}%</dd>

            <dt class="col-sm-8 dt-no-wrap">SVM F1 Score:</dt>
            <dd class="col-sm-2">{{ responseData.SVM_F1_Score}}</dd>

            <dt class="col-sm-8">SVM Mean Squared Error (MSE):</dt>
            <dd class="col-sm-2">{{ responseData.SVM_Mean_Squared_Error }}</dd>

            <dt class="col-sm-8">SVM Root Mean Squared Error (RMSE):</dt>
            <dd class="col-sm-2">{{ responseData.SVM_Root_Mean_Squared_Error }}</dd>
          </dl>
        </div>
      </div>
    </div>
    <div class="col d-flex flex-column align-items-center gap-2">
    <div class="row justify-content-center px-5 ">
        <div class="col-lg-12">
            <div class="alert alert-warning alert-dismissible fade show d-flex align-items-center justify-content-center" role="alert">
                <i class="bi bi-exclamation-triangle-fill me-2 fs-1 p-1 no-print"></i>
                <div>
                    <strong v-if="responseData.SVM_Probability <45">Low Chance of Cardiovascular Disease! {{ responseData.SVM_Probability }}%</strong>
                    <strong v-if="responseData.SVM_Probability >=45 ">High Chance of Cardiovascular Disease! {{ responseData.SVM_Probability }}%</strong>
                    <p class="mb-0" v-if="responseData.SVM_Probability <45">For further investigation of symptoms, the patient is recommended to be admitted to Emergency Room. </p>
                    <p class="mb-0" v-if="responseData.SVM_Probability >=45">For further specialist checking, the patient is recommended to be admitted for confinement. </p>
                  </div>
                <button type="button" class="btn-close ms-auto" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        </div>
    </div>
    <div>
        <button type="button" class="btn btn-outline-primary btn-lg px-5 mx-3 w-45 no-print fw-bold" @click="printPage" >Download Ticket</button>
        <button type="button" class="btn btn-outline-primary btn-lg px-5 mx-3 w-45 no-print fw-bold" @click="printPage" >Print Ticket</button>
    </div>
    <div class="mt-2 row">
      <div class="mb-3 form-floating">
        <input type="text" class="form-control" id="inputName" placeholder="Enter your Name" required>
        <label for="inputName">Patient Name:</label>
    </div>
    <div class="mb-3 form-floating">
        <input type="text" class="form-control" id="patientId" :value="randomNumber" disabled>
        <label for="patientId">Patient ID:</label>
    </div>
    </div>
    </div>

  </div>
    <h3 class="fw-bold mt-5 pt-5 no-print"> VISUALIZATION REPORT</h3>
    <div v-for="(url, index) in imageUrls" :key="index" class="crop no-print">
      <img :src="url" alt="Image" class="img-fluid img-thumbnail rounded mx-auto d-block"/>
    </div>
  </main>
</template>

<script>
import axios from 'axios'
import NavBar from '@/components/NavBar.vue'

export default {
  name: 'ResultView',
  components: {
    NavBar
  },
  data() {
    return {
      randomNumber: null,
      imageUrls: [],
      finalData: {
        age: '',
        gender: '',
        trestbps: '',
        history: '',
        cp: '',
        chol: '',
        fbs: '',
        restecg: '',
        thalach: '',
        thal: ''   
      },
      responseData: {
        SVM_Predicted_Class: '',
        SVM_Probability: '',
        SVM_Accuracy: '',
        SVM_Confusion_Matrix: '',
        SVM_Precision: '',
        SVM_Recall: '',
        SVM_F1_Score: '',
        SVM_Mean_Squared_Error: '',
        SVM_Root_Mean_Squared_Error: ''
      }
    }
  },
mounted() {
  this.finalData = JSON.parse(localStorage.getItem('finalInput'));
  this.responseData = JSON.parse(localStorage.getItem('responseData'));
  console.log(this.finalData)
  try {
  axios.get('http://127.0.0.1:5000/getImages')
    .then(response => {
      this.imageUrls = response.data.image_urls;
    })
    .catch(error => {
      console.error(error);
    });
} catch (error) {
  console.error(error);
}
},
methods: {
  generateRandomNumber() {
      this.randomNumber = Math.floor(Math.random() * 99999) + 100000;  // Generate a new random number between 0 and 100
    },
  printPage() {
    window.print()
  }
},
created() {
  this.generateRandomNumber();
}
}
</script>

<style scoped>
.crop {
  display: flex;
  justify-content: center;
  align-items: center;
  overflow: hidden;
  height: 100%;
}

.crop img {
  flex-shrink: 0;
  min-width: 115%;
  min-height: 110%;
}
.dt-no-wrap {
  white-space: nowrap;
  }
@media print {
  @page {
    margin: 20mm;
  }
  main {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    margin: 0;
    padding: 0;
  }
  .col {
    margin: 0;
    padding: 0;
  }
  .no-print {
    display: none;
  }
}

</style>