<template>
    <div>
      <h3>Prediction Values:</h3>
      <p>
        acousticness: {{ predictions.target_acousticness }} <br />
        danceability: {{ predictions.target_danceability }} <br />
        energy: {{ predictions.target_energy }} <br />
        instrumentalness: {{ predictions.target_instrumentalness }} <br />
        liveness: {{ predictions.target_liveness }} <br />
        speechiness: {{ predictions.target_speechiness }} <br />
        valence: {{ predictions.target_valence }} <br />
        seed_genres: {{ predictions.seed_genres }} <br />
        initialPoolSize: {{ resultSize.initialPoolSize }} <br />
        afterFilteringSize: {{ resultSize.afterFilteringSize }} <br />
        afterRelinkingSize: {{ resultSize.afterRelinkingSize }}
      </p>
    </div>
</template>

<script>
export default {
    props: ['predictions', 'resultSize'],
}
</script>

<style scoped>

</style>