const express = require('express')
const router = new express.Router()
const { getCategories } = require('../services/categories')

router.get('/api/categories', async (req, res) => {
    try {
        const categories = await getCategories()
        res.json(categories)
    } catch(e) {
        res.status(500).send(e)
    }
})

module.exports = router