<template>
    <div v-if="visible" class="modal fade show" tabindex="-1" aria-labelledby="exampleModalLabel" aria-modal="true" role="dialog" style="display:block">
      <div class="modal-dialog modal-dialog-centered">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title">Good Day! Here is your evaluation</h5>
            <button type="button" @click="closeModal" class="btn-close"></button>
          </div>
          <div class="modal-body">
            <p>Age: XX</p>
            <p>Gender: Male</p>
            <p>History of Heart Attack: Yes</p>
            <p>CVD Probability: Over **% HIGH CHANCE OF CVD</p>
          </div>
          <div class="modal-footer">
            <button type="button" @click="closeModal" class="btn btn-primary">Close</button>
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
    },
    methods: {
      closeModal() {
        this.$emit('update:visible', false);
      },
    },
    openModal() {
      this.modalVisible = true;
    },
  }
  </script>