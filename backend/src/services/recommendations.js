var rp = require('request-promise')
const util = require('util')
const exec = util.promisify(require('child_process').exec)
const path = require('path')
const spawn = require("child_process").spawn;
var { getAuthOptions } = require('./spotifyToken')

/**
 * Get artist seed from a given artist name 
 * @param {*} artistName 
 */
async function getArtistSeed(artistName) {
  var options = await getAuthOptions()
  options['url'] = 'https://api.spotify.com/v1/search?q=' + artistName + '&type=artist'
  const result = await rp.get(options)
  return result.artists.items[0].id
}

/**
 * Get recommendation for artist_seed.
 * @param {*} artistJson 
 */
async function getRecommendationsArtist(artistName) {
  const artistSeed = await getArtistSeed(artistName)

  const artistJson = {
      seed_artists: artistSeed,
      min_popularity: 60,
      market: 'DE',
      limit: 20
  }

  var options = await getAuthOptions()
  options['url'] = getUrlWithParameters(artistJson)

  const result = await rp.get(options)
  return extractDataFromSpotify(result.tracks)
}

async function downloadVideo(link) {
  try {
    const { stdout, stderr } = await exec ('youtube-dl -x --audio-format mp3 -o "upload/yt.%(ext)s" ' + link);
    return true
  } catch(err) {
    console.error(err)
    return false
  }
}

/**
 * 
 */
async function getRecommendationsFile(filename) {
  let predictedJson = await runPythonScript(filename)

  const recommendations = await getRecommendationsAi(predictedJson)

  return {
    "recommendations": recommendations,
    "predictions": predictedJson
  }
}

/**
 * Get recommendation for ai generated json data.
 * @param {*} aiJson 
 */
async function getRecommendationsAi(aiJson) {
  var options = await getAuthOptions()
  options['url'] = getUrlWithParameters(aiJson)

  const result = await rp.get(options)

  resultTracks = extractDataFromSpotify(result.tracks)

  resultSize = {
    "initialPoolSize": result.seeds[0].initialPoolSize,
    "afterFilteringSize": result.seeds[0].afterFilteringSize,
    "afterRelinkingSize": result.seeds[0].afterRelinkingSize
  }

  return {
    "resultTracks": resultTracks,
    "resultSize": resultSize
  }
}

/**
 * Adds parameter from given parameter object to url and returns
 * assembled string. 
 * @param {*} parameterObject 
 */
function getUrlWithParameters(parameterObject) {
  var url = new URL('https://api.spotify.com/v1/recommendations')

  const entries = Object.entries(parameterObject)
  for (const [key, value] of entries) {
    url.searchParams.append(key, value)
  }

  return url.href
}

/**
 * Extracts track information from spotify tracks object and returns
 * Array with readable information.
 * @param {*} tracks 
 */
function extractDataFromSpotify(tracks) {
  var artistsAlbumObject = []
  var i = 1

  tracks.forEach(function (tracks) {
    entry = {
      id: i,
      albumName: tracks.album.name,
      songName: tracks.name,
      artistName: tracks.artists[0].name
    }
    artistsAlbumObject.push(entry)
    i++
  })

  return artistsAlbumObject
}

/**
 * 
 * @param {*} filename 
 */
async function runPythonScript(filename) {
        var child = spawn(path.join(__dirname, '../misc/venv/bin/python'),[path.join(__dirname, '../misc/predict.py'), filename]);

        process.stdin.pipe(child.stdin)

        let predictedJson = "";
        for await (const chunk of child.stdout) {
            predictedJson += chunk
        }

        predictedJson = JSON.parse(predictedJson)
        return predictedJson
}

module.exports = {
  getRecommendationsArtist,
  getRecommendationsAi,
  getRecommendationsFile,
  downloadVideo
}