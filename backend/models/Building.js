const mongoose = require('mongoose');

const buildingSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
  },
  address: {
    street: String,
    city: String,
    state: String,
    zipCode: String,
    country: String,
  },
  coordinates: {
    lat: Number,
    lng: Number,
  },
  type: {
    type: String,
    enum: ['residential', 'commercial', 'industrial', 'institutional', 'other'],
  },
  specifications: {
    height: Number,
    floors: Number,
    constructionYear: Number,
    roofType: String,
    area: Number,
  },
  owner: {
    name: String,
    contact: String,
    email: String,
  },
  inspectionHistory: [{
    date: Date,
    surveyId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Survey',
    },
    findings: String,
  }],
}, { timestamps: true });

module.exports = mongoose.model('Building', buildingSchema);
