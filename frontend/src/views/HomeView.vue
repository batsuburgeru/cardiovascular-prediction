<template>
  <main class="home" style="overflow-x: hidden;">
    <NavBar/>

    <div class="container-fluid">
      <div class="row align-items-center justify-content-between">
        <div class="col-md-6">
          <div class="hero-desc font-monospace text-start px-5">
            <h1 class="fw-bolder">Cardiovascular Disease Prediction</h1>
            <p>Welcome to our CVD (Cardiovascular Disease) Prediction Tool, a web application tailored for Emergency Rooms with a focus on simplicity for non-techy users. Our interface allows people to swiftly assess the likelihood of a person experiencing a heart attack. Designed to streamline decision-making in high-pressure situations, our CVD web app provides crucial insights to support timely and informed medical interventions. Trust in our accessible platform to empower healthcare providers with actionable information, promoting efficient and effective patient care.</p>
            <router-link to="/prelim"><button class="btn btn-outline-info px-5 fw-bold" type="submit">Begin Assessment</button></router-link>
          </div>
        </div>
        <div class="col-md-6">
          <div class="hero-img" style="height: calc(100vh - 95px);">
            <img src="../assets/hero-page-doctor.jpg" alt="Image of Doctor" style="height: 100%; object-fit: cover;">
          </div>
        </div>
      </div>
    </div>
  </main>
</template>

<script>
// @ is an alias to /src
import NavBar from '@/components/NavBar.vue'

export default {
  name: 'HomeView',
  components: {
    NavBar
  }
}
</script>
