<template>
  <v-container>
    <StringInputForm v-on:getRecommendations="getRecommendations" :disabled="showProgressCircular" label="Youtube URL"></StringInputForm>
    <ProgressCircular v-if="showProgressCircular"></ProgressCircular>
    <v-container v-if="recommendationSongs.length > 0">
      <PredictionListing :predictions="predictions" :resultSize="resultSize"></PredictionListing>
      <RecommendationTable :recommendationSongs="recommendationSongs"></RecommendationTable>
    </v-container>
  </v-container>
</template>

<script>
import axios from "axios";
import StringInputForm from "./small/StringInputForm";
import PredictionListing from "./small/PredictionListing";
import ProgressCircular from "./small/ProgressCircular";
import RecommendationTable from "./small/RecommendationTable";

export default {
  name: "YoutubeSearch",
  components: {
    PredictionListing,
    StringInputForm,
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
    getRecommendations: async function(url) {
        this.flipShowProgressCircular()
        let response = await axios.get(
          "http://localhost:8080/api/recommendation?ytlink=" + url
        );
        this.flipShowProgressCircular()
        this.recommendationSongs = response.data.recommendations.resultTracks;
        this.resultSize = response.data.recommendations.resultSize;
        this.predictions = response.data.predictions;
    },
    flipShowProgressCircular: function() {
      this.showProgressCircular = !this.showProgressCircular;
    }
  }
};
</script>

<style scoped>
</style>