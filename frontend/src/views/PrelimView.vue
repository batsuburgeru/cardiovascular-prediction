<template>
  <main class="home m-2" style="overflow-x: hidden;">
    <NavBar/>
    <ModalPrelim :visible="modalVisible" @update:visible="modalVisible = $event"></ModalPrelim>
    <div class="prelim container-fluid align-items-center" style="height: calc(100vh - 95px); background-color: white;" >
      <div class="row justify-content-center p-5">
        <div class="col-lg-5 text-start p-5" style="background-color: #E5FCFF;">
          <h2>Preliminary Test</h2>
          <form action="">
            <div class="mb-3 ">
                <label for="exampleInputEmail1" class="form-label">Age</label>
                <input type="integer" class="form-control shadow bg-body rounded" id="inputAge" aria-describedby="ageHelp">
              </div>
            <div class="mb-3">
              <label for="gender" class="form-label">Gender</label>
              <div class="form-check form-check-inline d-block">
                <input class="form-check-input" type="radio" name="gender" id="inlineRadio1" value="option1">
                <label class="form-check-label" for="inlineRadio1">Male</label>
              </div>
              <div class="form-check form-check-inline d-block">
                <input class="form-check-input" type="radio" name="gender" id="inlineRadio2" value="option2">
                <label class="form-check-label" for="inlineRadio2">Female</label>
              </div>
            </div>
            <div class="mb-3">
                <label for="exampleInputEmail1" class="form-label">Resting Blood Pressure</label>
                <input type="integer" class="form-control shadow bg-body rounded" id="inputAge" aria-describedby="ageHelp">
              </div>
              <div>
                <label for="" class="form-label">History of Cardiovascular Disease?</label>
                <div class="form-check form-check-inline d-block">
                  <input class="form-check-input" type="radio" name="gender" id="inlineRadio1" value="option1">
                  <label class="form-check-label" for="inlineRadio1">Yes</label>
                </div>
                <div class="form-check form-check-inline d-block">
                  <input class="form-check-input" type="radio" name="gender" id="inlineRadio2" value="option2">
                  <label class="form-check-label" for="inlineRadio2">No</label>
                </div>
              </div>
              <div class="mb-3">
                <label for="exampleInputEmail1" class="form-label">Chest Paint Type (CP)</label>
                <input type="integer" class="form-control shadow bg-body rounded" id="inputAge" aria-describedby="ageHelp">
              </div>
              <div class="d-grid gap-2">
                <button class="btn btn-primary" type="button" @click="openModal">Submit</button>
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

export default {
  name: 'PrelimView',
  components: {
    NavBar, ModalPrelim
  },
  data() {
    return {
      modalVisible: false,
    };
  },
  methods: {
    openModal() {
      this.modalVisible = true;
    },
  },
}
</script>