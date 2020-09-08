var rp = require('request-promise')
var client_id = process.env.CLIENT_ID
var client_secret = process.env.CLIENT_SECRET
var bearerToken = null

var authOptions = {
  url: 'https://accounts.spotify.com/api/token',
  headers: {
    'Authorization': 'Basic ' + (Buffer.from(client_id + ':' + client_secret).toString('base64'))
  },
  form: {
    grant_type: 'client_credentials'
  },
  json: true
};

async function getAuthOptions() {
  //TODO validate if token is older than 1 hour
  if (bearerToken === null) {
    const result = await rp.post(authOptions)
    bearerToken = result.access_token
  }

  return options = {
    json: true,
    headers: {
      'Authorization': 'Bearer ' + bearerToken
    }
  }
}

module.exports = {
    getAuthOptions
}