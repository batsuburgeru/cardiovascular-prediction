<template>
<nav class="navbar bg-body-tertiary py-4">
  <div class="container-fluid">
    <router-link to="/" class="navbar-brand px-5 fw-bolder" style="color: #5DC9E0;">B P M M</router-link>
    <form class="d-flex" role="search">
        <router-link to="/prelim"><button class="btn btn-outline-info px-5 fw-bold" type="submit">CVD Prediction</button></router-link>
    </form>
  </div>
</nav>
</template>

<script>
export default {
  name: 'NavBar',
  props: {
    msg: String
  }
}
</script>

<style scoped>
nav {
  background-color: white;
}
</style>