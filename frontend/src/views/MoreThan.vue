<template>
    <main class="home m-2" style="overflow-x: hidden;">
      <NavBar/>
      <div class="prelim container-fluid align-items-center" style="height: calc(100vh - 95px); background-color: white;" >
        <MoreThanModal/>
        <div class="row justify-content-center p-5">
          <div class="col-lg-5 text-start p-5" style="background-color: #E5FCFF;">
            <h2>Additional Information</h2>
            <form action="">
              <div class="mb-3 ">
                  <label for="exampleInputEmail1" class="form-label">Cholesterol (chol)</label>
                  <input type="integer" class="form-control shadow bg-body rounded" id="inputAge" aria-describedby="ageHelp">
                </div>
              <div class="mb-3">
                  <label for="exampleInputEmail1" class="form-label">Fasting Blood Sugar (fbs)</label>
                  <input type="integer" class="form-control shadow bg-body rounded" id="inputAge" aria-describedby="ageHelp">
                </div>
                <div class="mb-3">
                  <label for="exampleInputEmail1" class="form-label">Resting Electrocardiographic Result (restecg)</label>
                  <input type="integer" class="form-control shadow bg-body rounded" id="inputAge" aria-describedby="ageHelp">
                </div>
                <div class="mb-3 ">
                  <label for="exampleInputEmail1" class="form-label">Maximum Heart Rate Achieved (thalac)</label>
                  <input type="integer" class="form-control shadow bg-body rounded" id="inputAge" aria-describedby="ageHelp">
                </div>
                <div class="mb-3 ">
                  <label for="exampleInputEmail1" class="form-label">Thalium Stress Test Result (thal)</label>
                  <input type="integer" class="form-control shadow bg-body rounded" id="inputAge" aria-describedby="ageHelp">
                </div>
                <div class="d-grid gap-2">
                  <button class="btn btn-primary" type="button">Submit</button>
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
import MoreThanModal from '@/components/MoreThanModal.vue'

export default {
  name: 'MoreThanView',
  components: {
    NavBar,
    MoreThanModal
  }
}
</script>