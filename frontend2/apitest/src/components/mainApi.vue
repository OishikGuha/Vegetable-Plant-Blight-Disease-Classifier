<template>
  <label for="file">
    <input
      type="file"
      name="file"
      id="file"
      @change="handleFileUpload($event)"
    />
  </label>
  <button type="submit" @click="SubmitFile()">Submit</button>
  <p class="result">Prediction: {{ prediction }}</p>
  <p class="result">Confidence: {{ confidence }}</p>
</template>

<script>
import axios from "axios";

export default {
  name: "mainAPI",
  data() {
    return {
      response: "",
      file: "",
      prediction: "",
      confidence: "",
    };
  },
  methods: {
    Ping() {
      axios.get("http://localhost:8000/ping").then((response) => {
        this.response = response["data"];
        console.log(response["data"]);
      });
    },
    handleFileUpload(event) {
      this.file = event.target.files[0];
    },
    async SubmitFile() {
      console.log(this.file.length);

      let formData = new FormData();
      formData.append("file", this.file);

      await axios
        .post("http://localhost:8000/predict", formData)
        .then((response) => {
          console.log(response);
          this.prediction = response["data"]["class"];
          this.confidence = response["data"]["confidence"];
        });
    },
  },
  mounted() {
    this.Ping();
  },
};
</script>