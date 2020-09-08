<template>
  <v-container>
    <StringInputForm v-on:getRecommendations="getRecommendations" :disabled="showProgressCircular" label="Artist"></StringInputForm>
    <RecommendationTable v-if="!showProgressCircular && recommendationSongs.length > 0" :recommendationSongs="recommendationSongs">
    </RecommendationTable>
    <ProgressCircular v-if="showProgressCircular"></ProgressCircular>
  </v-container>
</template>

<script>
import axios from "axios";
import RecommendationTable from "./small/RecommendationTable";
import ProgressCircular from "./small/ProgressCircular";
import StringInputForm from "./small/StringInputForm";

export default {
  name: "ArtistSearch",
  components: {
    RecommendationTable,
    ProgressCircular,
    StringInputForm
  },
  data() {
    return {
      recommendationSongs: {},
      showProgressCircular: false,
    };
  },
  methods: {
    getRecommendations: async function(artist) {
      this.flipShowProgressCircular()
      let response = await axios.get(
        "http://localhost:8080/api/recommendation?artist=" + artist
      );
      this.recommendationSongs = response.data;
      this.flipShowProgressCircular()
    },
    flipShowProgressCircular: function() {
      this.showProgressCircular = !this.showProgressCircular;
    }
  }
};
</script>

<style scoped>
</style>