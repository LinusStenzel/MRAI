var rp = require('request-promise');
var { getAuthOptions } = require('./spotifyToken')

async function getCategories() {
  var options = await getAuthOptions()
  options['url'] = 'https://api.spotify.com/v1/recommendations/available-genre-seeds'
  return result = await rp.get(options)
}

module.exports = {
  getCategories
}