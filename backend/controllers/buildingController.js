const Building = require('../models/Building');

// Get all buildings
exports.getAllBuildings = async (req, res) => {
  try {
    const buildings = await Building.find().sort({ createdAt: -1 });
    res.json(buildings);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// Get single building
exports.getBuilding = async (req, res) => {
  try {
    const building = await Building.findById(req.params.id)
      .populate('inspectionHistory.surveyId');
    if (!building) {
      return res.status(404).json({ message: 'Building not found' });
    }
    res.json(building);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// Create building
exports.createBuilding = async (req, res) => {
  const building = new Building(req.body);
  try {
    const newBuilding = await building.save();
    res.status(201).json(newBuilding);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
};

// Update building
exports.updateBuilding = async (req, res) => {
  try {
    const building = await Building.findByIdAndUpdate(
      req.params.id,
      req.body,
      { new: true }
    );
    if (!building) {
      return res.status(404).json({ message: 'Building not found' });
    }
    res.json(building);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
};

// Delete building
exports.deleteBuilding = async (req, res) => {
  try {
    const building = await Building.findByIdAndDelete(req.params.id);
    if (!building) {
      return res.status(404).json({ message: 'Building not found' });
    }
    res.json({ message: 'Building deleted successfully' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};
