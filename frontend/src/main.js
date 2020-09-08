import Vue from 'vue'
import App from './App.vue'
import VueRouter from 'vue-router'
import vuetify from './plugins/vuetify';
import ArtistSearch from './components/ArtistSearch'
import FileSearch from './components/FileSearch'
import YoutubeSearch from './components/YoutubeSearch'

Vue.config.productionTip = false
Vue.use(VueRouter)

const routes = [
  { path: '/', component: ArtistSearch},
  { path: '/file', component: FileSearch},
  { path: '/youtube', component: YoutubeSearch}
]

const router = new VueRouter({
  routes
})

new Vue({
  router,
  vuetify,
  render: h => h(App)
}).$mount('#app')
