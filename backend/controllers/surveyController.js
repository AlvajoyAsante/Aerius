const Survey = require('../models/Survey');

// Get all surveys
exports.getAllSurveys = async (req, res) => {
  try {
    const surveys = await Survey.find()
      .populate('droneId')
      .populate('buildingId')
      .sort({ createdAt: -1 });
    res.json(surveys);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// Get single survey
exports.getSurvey = async (req, res) => {
  try {
    const survey = await Survey.findById(req.params.id)
      .populate('droneId')
      .populate('buildingId');
    if (!survey) {
      return res.status(404).json({ message: 'Survey not found' });
    }
    res.json(survey);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// Create survey
exports.createSurvey = async (req, res) => {
  const survey = new Survey(req.body);
  try {
    const newSurvey = await survey.save();
    res.status(201).json(newSurvey);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
};

// Update survey
exports.updateSurvey = async (req, res) => {
  try {
    const survey = await Survey.findByIdAndUpdate(
      req.params.id,
      req.body,
      { new: true }
    );
    if (!survey) {
      return res.status(404).json({ message: 'Survey not found' });
    }
    res.json(survey);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
};

// Delete survey
exports.deleteSurvey = async (req, res) => {
  try {
    const survey = await Survey.findByIdAndDelete(req.params.id);
    if (!survey) {
      return res.status(404).json({ message: 'Survey not found' });
    }
    res.json({ message: 'Survey deleted successfully' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};
