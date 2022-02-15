<template>
  <div class="main">
    <h1>Potato/Tomato Blight Disease Classifier AI</h1>
    <div class="mainAPI">
      <mainAPI />
    </div>
    <p>-Made by Oishik Guha aka. roboticol</p>
  </div>
</template>

<script>
import mainAPI from "./components/mainApi.vue";

export default {
  name: "App",
  components: { mainAPI },
};
</script>

<style>
body {
  background-image: url("background.png");
  background-position: center;
  background-repeat: no-repeat;
  background-size: cover;
}

h1 {
  position: absolute;
  color: springgreen;
  font-family: Helvetica;
  text-align: center;
  font-weight: bolder;
  outline-style: dashed;
  margin-left: 40%;
  margin-right: 40%;
  transform: translateY(-150%);
}

.mainAPI {
  margin-top: 20%;
}

p {
  font-family: Helvetica, sans-serif;
  font-weight: bold;
  text-align: center;
  margin-top: 20%;
  display: block;
  color: rgb(255, 255, 255);
}

label {
  font-family: Helvetica, sans-serif;
  font-weight: bolder;
  color: rgb(255, 255, 255);
  display: block;
  align-content: center;
  text-align: center;
}

button {
  font-family: Helvetica, sans-serif;
  font-weight: bolder;
  position: absolute;
  transform: translateX(1500%);
  display: block;
  align-content: center;
  text-align: center;
}

.result {
  transform: translateY(250%);
  margin-top: -1%;
  color: rgb(139, 179, 253);
}
</style>
