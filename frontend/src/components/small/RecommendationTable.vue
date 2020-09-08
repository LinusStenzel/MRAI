<template>
    <div class="recommendationTable">
      <v-simple-table>
        <thead>
          <th>#</th>
          <th>Interpret</th>
          <th>Song</th>
          <th>Album</th>
        </thead>
        <tbody>
          <tr v-for="album in recommendationSongs" v-bind:key="album.id">
            <th>{{ album.id }}</th>
            <td>{{ album.artistName }}</td>
            <td>{{ album.songName }}</td>
            <td>{{ album.albumName }}</td>
          </tr>
        </tbody>
      </v-simple-table>
    </div>
</template>

<script>
export default {
    props: ['recommendationSongs'],
}
</script>

<style scoped>

</style>