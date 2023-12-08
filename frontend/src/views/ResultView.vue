<template>
  <main>
    <NavBar/>
    <div class="row align-items-start px-5 mx-5">
      <h2> Result </h2>
    <div class="col">
      <div class="mx-auto" style="width: 75%;">
        <h3>Input</h3>
        <div v-if="finalData" class="row text-start">
          <dl class="row">
            <dt class="col-sm-10 dt-no-wrap">Age:</dt>
            <dd class="col-sm-2">{{ finalData.age }}</dd>

            <dt class="col-sm-10 dt-no-wrap">Gender:</dt>
            <dd class="col-sm-2">{{ finalData.gender }}</dd>

            <dt class="col-sm-10 dt-no-wrap">Resting Blood Pressure:</dt>
            <dd class="col-sm-2">{{ finalData.trestbps }}</dd>

            <dt class="col-sm-10 dt-no-wrap">History of Heart Disease:</dt>
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
        <h3>SVM Result</h3>
        <div v-if="responseData" class="row text-start">
          <dl class="row">
            <dt class="col-sm-10 dt-no-wrap">SVM Predicted Class:</dt>
            <dd class="col-sm-2">{{ responseData.SVM_Predicted_Class }}</dd>

            <dt class="col-sm-10 dt-no-wrap">SVM Probability:</dt>
            <dd class="col-sm-2">{{ responseData.SVM_Probability }}</dd>

            <dt class="col-sm-10 dt-no-wrap">SVM Accuracy:</dt>
            <dd class="col-sm-2">{{ responseData.SVM_Accuracy }}</dd>

            <dt class="col-sm-10 dt-no-wrap">SVM Confusion Matrix:</dt>
            <dd class="col-sm-2">{{ responseData.SVM_Confusion_Matrix[0] }}</dd>
            <dt class="col-sm-10 dt-no-wrap"></dt>
            <dd class="col-sm-2">{{ responseData.SVM_Confusion_Matrix[1] }}</dd>

            <dt class="col-sm-10 dt-no-wrap">SVM Precision:</dt>
            <dd class="col-sm-2">{{ responseData.SVM_Precision}}</dd>

            <dt class="col-sm-10 dt-no-wrap">SVM Recall:</dt>
            <dd class="col-sm-2">{{ responseData.SVM_Recall }}</dd>

            <dt class="col-sm-10 dt-no-wrap">SVM F1 Score:</dt>
            <dd class="col-sm-2">{{ responseData.SVM_F1_Score}}</dd>

            <dt class="col-sm-10">SVM Mean Squared Error (MSE):</dt>
            <dd class="col-sm-2">{{ responseData.SVM_Mean_Squared_Error }}</dd>

            <dt class="col-sm-10">SVM Root Mean Squared Error (RMSE):</dt>
            <dd class="col-sm-2">{{ responseData.SVM_Root_Mean_Squared_Error }}</dd>

          </dl>
        </div>
      </div>
    </div>
    <div class="col">
      One of three columns
    </div>
  </div>
    <div v-for="(url, index) in imageUrls" :key="index" class="crop">
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
        SVM_Root_Mean_Squared_Error: '',



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
}
}
</script>

<style scoped>
.crop {
  margin: 3rem 0;
  display: flex;
  justify-content: center;
  align-items: center;
  overflow: hidden;
  height: 4550px; /* Adjust as needed */
}

.crop img {
  flex-shrink: 0;
  min-width: 100%;
  min-height: 100%;
}
.dt-no-wrap {
  white-space: nowrap;
  }
</style>