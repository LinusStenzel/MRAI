const express = require('express')
var cors = require('cors')
const spotifyCategoriesRouter = require('./routers/spotifyCategories')
const spotifyRecommendationsRouter = require('./routers/spotifyRecommendations')
const port = process.env.PORT

const app = new express()
app.use(cors())
app.use(express.json())
app.use(spotifyCategoriesRouter)
app.use(spotifyRecommendationsRouter)

app.listen(port, () => {
  console.log('Server is running on port ' + port)
})
