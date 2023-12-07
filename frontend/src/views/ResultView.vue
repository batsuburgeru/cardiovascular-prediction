<template>
  <main>
    <NavBar/>
    <div>
      
    </div>

  </main>

      <!--<div v-for="(url, index) in imageUrls" :key="index">
      <img :src="url" alt="Image" class="img-fluid img-thumbnail rounded mx-auto d-block"/>
    </div>-->
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
      imageUrls: []
    }
  },
mounted() {
  this.finalData = JSON.parse(localStorage.getItem('finalData'));
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
