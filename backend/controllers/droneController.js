const Drone = require('../models/Drone');

// Get all drones
exports.getAllDrones = async (req, res) => {
  try {
    const drones = await Drone.find().sort({ createdAt: -1 });
    res.json(drones);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// Get single drone
exports.getDrone = async (req, res) => {
  try {
    const drone = await Drone.findById(req.params.id);
    if (!drone) {
      return res.status(404).json({ message: 'Drone not found' });
    }
    res.json(drone);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// Create drone
exports.createDrone = async (req, res) => {
  const drone = new Drone(req.body);
  try {
    const newDrone = await drone.save();
    res.status(201).json(newDrone);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
};

// Update drone
exports.updateDrone = async (req, res) => {
  try {
    const drone = await Drone.findByIdAndUpdate(
      req.params.id,
      req.body,
      { new: true }
    );
    if (!drone) {
      return res.status(404).json({ message: 'Drone not found' });
    }
    res.json(drone);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
};

// Delete drone
exports.deleteDrone = async (req, res) => {
  try {
    const drone = await Drone.findByIdAndDelete(req.params.id);
    if (!drone) {
      return res.status(404).json({ message: 'Drone not found' });
    }
    res.json({ message: 'Drone deleted successfully' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};
