const express = require('express')
const multer = require('multer')

const router = new express.Router()
const { 
    getRecommendationsAi,
    getRecommendationsArtist,
    getRecommendationsFile,
    downloadVideo
} = require('../services/recommendations')

var storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'upload/')
    },
    filename: function (req, file, cb) {
        cb(null, file.fieldname + '-' + Date.now() + '.mp3')
    }
})

var upload = multer({ storage: storage })

router.get('/api/recommendation', async (req, res) => {
    try {
        if (req.query.artist) { // artist
            const recommendations = await getRecommendationsArtist(req.query.artist)
            res.json(recommendations)
        } else if (req.query.ytlink) { // yt link
            const success = await downloadVideo(req.query.ytlink)
            if (success) {
                const recommendations = await getRecommendationsFile('yt.mp3')
                res.json(recommendations)
            } 
        } else {
            throw new Exception('error')
        }
    } catch(e) {
        res.json(e)
    }
})

router.post('/api/recommendation', upload.single('mp3File'), async (req, res) => {
    try {
        const recommendations = await getRecommendationsFile(req.file.filename)
        res.json(recommendations)
    } catch(e) {
        res.json(e)
    }
})

module.exports = router