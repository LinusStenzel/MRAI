# projektstudie_music_ai

## TODO
* Gegenüberstellung was unser Netz aus einem mp3 file predicted und was Spotify predicted?

## prerequisites
* [node](https://nodejs.org/en/download/) 12.13.0 
* [npm](https://docs.npmjs.com/cli/install) 6.2.1
* [yarn](https://yarnpkg.com/lang/en/docs/install/) 1.19.1
* [python3](https://www.python.org/downloads/) 3.7.4
* [pip](https://pip.pypa.io/en/stable/installing/) 19.2.3
* [virtualenv](https://docs.python.org/3/library/venv.html) 16.1.0 `pip install virtualenv`
* [FFmpeg](http://www.google.de) 
* [youtube-dl](http://www.google.de)

## install app
### backend
* install npm dependencies with: `npm install` from backend directory
* create virtual python env in backend/src/misc: `virtualenv venv`
* install pip dependencies: `pip install -r requirements.txt` from backend/src/misc
### frontend
* install yarn dependencies with: `yarn install` from frontend directory

## run app
### backend
* run in dev mode: `npm run dev`
### frontend
* run in dev mode: `yarn serve`

## use
### backend api
* request all categories from spotify api: http://127.0.0.1:8080/api/categories
* get example recommendation from spotify api for an artist: http://127.0.0.1:8080/api/recommendations?artist=$ARTIST
* post http request mp3 file which should be stored on the server: http://127.0.0.1:8080/api/recommendations
### frontend
* Vue app to query recommendation from spotify api: http://127.0.0.1:8081
* Upload a mp3 to node server: http://127.0.0.1:8081/file

## Parameter for recommendation api from spotify
* acousticness float
* danceability float
* energy float
* instrumentalness float
* liveness float
* speechiness float
* valence float

## Seeds
* seed_artists
* seed_genres
* seed_tracks

## Genres
```json
{
  "genres": [
    "classical",
    "electronic",
    "folk",
    "hip-hop",
    "jazz",
    "pop",
    "rock"
  ]
}
```
