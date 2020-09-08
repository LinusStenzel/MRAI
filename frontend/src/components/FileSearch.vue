<template>
  <v-container>
    <FileUploadForm v-on:uploadFile="uploadFile"></FileUploadForm>
    <ProgressCircular v-if="showProgressCircular"></ProgressCircular>
    <v-container v-if="recommendationSongs.length > 0">
      <PredictionListing :predictions="predictions" :resultSize="resultSize"></PredictionListing>
      <RecommendationTable :recommendationSongs="recommendationSongs"></RecommendationTable>
    </v-container>
  </v-container>
</template>

<script>
import axios from "axios";
import PredictionListing from "./small/PredictionListing";
import FileUploadForm from "./small/FileUploadForm";
import ProgressCircular from "./small/ProgressCircular";
import RecommendationTable from "./small/RecommendationTable";

export default {
  name: "FileSearch",
  components: {
    PredictionListing,
    FileUploadForm,
    ProgressCircular,
    RecommendationTable
  },
  data() {
    return {
      showProgressCircular: false,
      recommendationSongs: {},
      predictions: {},
      resultSize: {}
    };
  },
  methods: {
    uploadFile: async function(file) {
      if (file) {
        this.flipShowProgressCircular();
        let formData = new FormData();

        formData.append("mp3File", file, file.name);
        let response = await axios.post(
          "http://localhost:8080/api/recommendation",
          formData
        );
        if (response) {
        this.flipShowProgressCircular();
          this.recommendationSongs = response.data.recommendations.resultTracks;
          this.resultSize = response.data.recommendations.resultSize;
          this.predictions = response.data.predictions;
        }
      }
    },
    flipShowProgressCircular: function() {
      this.showProgressCircular = !this.showProgressCircular;
    }
  }
};
</script>

<style scoped>
</style>