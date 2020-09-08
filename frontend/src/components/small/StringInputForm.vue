<template>
    <div class="StringInputForm">
        <v-form ref="form">
          <v-row cols="12" md="4">
            <v-col cols="6" md="4">
              <v-text-field v-model="artist" :label="label"></v-text-field>
            </v-col>
            <v-col cols="6" md="4">
              <v-btn color="success" class="mr-4" :disabled="disabled" v-on:click="$emit('getRecommendations', artist)">Search for recommendation</v-btn>
            </v-col>
          </v-row>
        </v-form>
    </div>
</template>

<script>
export default {
    data() {
        return {
            artist: ""
        }
    },
    props: ['label', 'disabled']
}
</script>

<style scoped>

</style>