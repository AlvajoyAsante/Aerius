const mongoose = require('mongoose');

const droneSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
  },
  model: {
    type: String,
    required: true,
  },
  manufacturer: String,
  serialNumber: {
    type: String,
    unique: true,
  },
  specifications: {
    maxAltitude: Number,
    maxFlightTime: Number,
    cameraResolution: String,
    weight: Number,
  },
  status: {
    type: String,
    enum: ['available', 'in-use', 'maintenance', 'retired'],
    default: 'available',
  },
  lastMaintenance: Date,
  purchaseDate: Date,
}, { timestamps: true });

module.exports = mongoose.model('Drone', droneSchema);
