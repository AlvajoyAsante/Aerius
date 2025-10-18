const mongoose = require('mongoose');

const surveySchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
  },
  location: {
    address: String,
    coordinates: {
      lat: Number,
      lng: Number,
    },
  },
  droneId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Drone',
  },
  buildingId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Building',
  },
  date: {
    type: Date,
    default: Date.now,
  },
  status: {
    type: String,
    enum: ['planned', 'in-progress', 'completed', 'cancelled'],
    default: 'planned',
  },
  notes: String,
  images: [String],
  data: {
    altitude: Number,
    weather: String,
    temperature: Number,
    windSpeed: Number,
  },
}, { timestamps: true });

module.exports = mongoose.model('Survey', surveySchema);
