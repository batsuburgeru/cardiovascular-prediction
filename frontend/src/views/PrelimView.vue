<template>
  <main class="home m-2" style="overflow-x: hidden;">
    <NavBar/>
    <ModalPrelim :visible="modalVisible" :responseData="responseData" @update:visible="modalVisible = $event"></ModalPrelim>
    <div class="prelim container-fluid align-items-center" style="height: calc(100vh - 95px); background-color: white;" >
      <div class="row justify-content-center p-5">
        <div class="col-lg-5 text-start p-5" style="background-color: #E5FCFF;">
          <h2>Preliminary Test</h2>
          <form action="">
            <div class="mb-3 form-floating">
              <input type="age" class="form-control" id="inputAge" placeholder="e.g 60" v-model="prelimData.age">
                <label for="inputAge">Age</label>
              </div>
            <div class="mb-3">
              <label class="form-label">Gender</label>
              <div class="form-check form-check-inline d-block">
                <input class="form-check-input" type="radio" name="gender" id="inputGender1" value="1" v-model="prelimData.gender" >
                <label class="form-check-label" for="inputGender1">Male</label>
              </div>
              <div class="form-check form-check-inline d-block">
                <input class="form-check-input" type="radio" name="gender" id="inputGender0" value="0" v-model="prelimData.gender">
                <label class="form-check-label" for="inputGender0">Female</label>
              </div>
            </div>
            <div class="mb-3 form-floating">
                <input type="number" class="form-control shadow bg-body rounded" id="inputTrestbps" placeholder="e.g 145" aria-describedby="Resting Blood Pressure" v-model="prelimData.trestbps">
                <label for="inputTrestbps">Resting Blood Pressure</label>
              </div>
              <div>
                <label class="form-label">History of Cardiovascular Disease?</label>
                <div class="form-check form-check-inline d-block">
                  <input class="form-check-input" type="radio" name="history" id="inputHistory1" value="1" v-model="prelimData.history">
                  <label class="form-check-label" for="inputHistory1">Yes</label>
                </div>
                <div class="form-check form-check-inline d-block">
                  <input class="form-check-input" type="radio" name="history" id="inputHistory0" value="0" v-model="prelimData.history">
                  <label class="form-check-label" for="inputHistory0">No</label>
                </div>
              </div>
              <div class="mb-3 form-floating">
                <input type="number" class="form-control shadow bg-body rounded" id="inputCP"  placeholder="e.g 4" aria-describedby="cp" v-model="prelimData.cp">
                <label for="inputCP">Chest Paint Type (CP)</label>
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
import ModalPrelim from '@/components/ModalPrelim.vue'
import axios from 'axios';
import { Modal } from 'bootstrap';

export default {
  name: 'PrelimView',
  components: {
    NavBar, ModalPrelim
  },
  data() {
    return {
      responseData: null,
      prelimData: {
      age: '',
      gender: '',
      trestbps: '',
      history: '',
      cp: ''
      },
      modalVisible: false,
      
    };
  },
  methods: {
    async handleSubmit() {
      var myModal = new Modal(document.getElementById('exampleModalToggle'), {});
      myModal.show();
      this.modalVisible = true;
      try {
        const url = 'http://127.0.0.1:5000/prelim';
        const response = await axios.post(url, {
          age: this.prelimData.age,
          gender: this.prelimData.gender,
          trestbps: this.prelimData.trestbps,
          history: this.prelimData.history,
          cp: this.prelimData.cp
        });
        if (response.status === 200) {
          this.responseData = response.data;
          localStorage.setItem('prelimData', JSON.stringify(this.prelimData))
          console.log('Request Successful:', response.data);
        } else {
          console.error('Request Failed:', response.data);
          alert('Request Failed:' + response.data.message);
        }
      } catch (error) {
        console.error('Request Failed:', error.message);
      }
    },
    
  },
}
</script>
