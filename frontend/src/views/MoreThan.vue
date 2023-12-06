<template>
    <main class="home m-2" style="overflow-x: hidden;">
      <NavBar/>
      <div class="prelim container-fluid align-items-center" style="height: calc(100vh - 95px); background-color: white;" >
        <MoreThanAlert/>
        <ResultModal :visible="modalVisible" :responseData="responseData" @update:visible="modalVisible = $event"></ResultModal>
        <div class="row justify-content-center p-5">
          <div class="col-lg-5 text-start p-5" style="background-color: #E5FCFF;">
            <h2>Additional Information</h2>
            <form action="">
              <div class="mb-3 form-floating">
                  <input type="number" class="form-control shadow bg-body rounded" id="inputChol" placeholder="e.g 60" v-model="moreThanData.chol"> 
                  <label for="inputChol">Cholesterol</label>
                </div>
              <div class="mb-3 form-floating">
                  <input type="number" class="form-control shadow bg-body rounded" id="inputFBS" placeholder="e.g 60" v-model="moreThanData.fbs">
                  <label for="inputFBS">Fasting Blood Sugar (fbs)</label>
                </div>
                <div class="mb-3 form-floating">
                  <input type="number" class="form-control shadow bg-body rounded" id="inputrestecg" placeholder="e.g 60" v-model="moreThanData.restecg">
                  <label for="inputRestecg">Resting Electrocardiographic Result (restecg)</label>
                </div>
                <div class="mb-3 form-floating">
                  <input type="number" class="form-control shadow bg-body rounded" id="inputThalac" placeholder="e.g 60" v-model="moreThanData.thalach">
                  <label for="inputThalac">Maximum Heart Rate Achieved (thalac)</label>
                </div>
                <div class="mb-3 form-floating">
                  <input type="number" class="form-control shadow bg-body rounded" id="inputThal" placeholder="e.g 60" v-model="moreThanData.thal">
                  <label for="inputThal">Thalassemia Type (thal)</label>
                </div>
                <div class="d-grid gap-2">
                  <button class="btn btn-primary" type="button" @click="handleSubmit">Submit</button>
                </div>
              </form>
            </div>
            </div>
          </div>
      </main>
  </template>

<script>
// @ is an alias to /src
import NavBar from '@/components/NavBar.vue'
import MoreThanAlert from '@/components/MoreThanAlert.vue'
import axios from 'axios';
import ResultModal from '@/components/ResultModal.vue'
import { Modal } from 'bootstrap';

export default {
  name: 'MoreThanView',
  components: {
    NavBar, MoreThanAlert, ResultModal
  },
  data() {
    return {
      responseData: null,
      modalVisible: false,

      prelimData: {
      age: '',
      gender: '',
      trestbps: '',
      history: '',
      cp: ''
      },
      moreThanData: {
        chol: '',
        fbs: '',
        restecg: '',
        thalach: '',
        thal: ''   
      },
    };
  },
  mounted() {
  this.prelimData = JSON.parse(localStorage.getItem('prelimData'));
},
methods: {
  async handleSubmit() {
    var myModal = new Modal(document.getElementById('exampleModalToggle'), {});
      myModal.show();
      this.modalVisible = true;
    try {
      const url = 'http://127.0.0.1:5000/moreThan'
      const response = await axios.post(url, {
        age: this.prelimData.age,
        gender: this.prelimData.gender,
        trestbps: this.prelimData.trestbps,
        history: this.prelimData.history,
        cp: this.prelimData.cp,
        chol: this.moreThanData.chol,
        fbs: this.moreThanData.fbs,
        restecg: this.moreThanData.restecg,
        thalach: this.moreThanData.thalach,
        thal: this.moreThanData.thal
      });
      console.log('Response Data:', response.data);
      if(response.status === 200) {
        this.responseData = response.data;
        const finalInput = {...this.prelimData, ...this.moreThanData};
        localStorage.setItem('finalInput', JSON.stringify(finalInput))
        console.log('Request Successful:', response.data);
      }
      else {
        console.error('Request Failed:', response.data);
        alert('Request Failed' + response.data.message)
      }
    } catch (error) {
      console.error('Request Failed:', error.message)
    }
  }
}
}
</script> 