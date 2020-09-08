<template>
    <div class="FileUploadForm">
        <v-form ref="form">
          <v-row cols="12" md="4">
            <v-col cols="6" md="4">
              <v-file-input accept=".mp3" label="File input" v-model="file"></v-file-input>
            </v-col>
            <v-col cols="6" md="4">
              <v-btn color="success" class="mr-4" v-on:click="$emit('uploadFile', file)">Upload file</v-btn>
            </v-col>
          </v-row>
        </v-form>
    </div>
</template>

<script>
export default {
    data() {
        return {
            file: null
        }
    }    
}
</script>

<style scoped>

</style>