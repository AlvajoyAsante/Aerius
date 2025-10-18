const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const mongoose = require('mongoose');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB Connection
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/aerius';
mongoose.connect(MONGODB_URI)
.then(() => console.log('MongoDB connected successfully'))
.catch((err) => console.log('MongoDB connection error:', err));

// Routes
const surveyRoutes = require('./routes/surveys');
const droneRoutes = require('./routes/drones');
const buildingRoutes = require('./routes/buildings');

app.use('/api/surveys', surveyRoutes);
app.use('/api/drones', droneRoutes);
app.use('/api/buildings', buildingRoutes);

// Root endpoint
app.get('/', (req, res) => {
  res.json({ message: 'Welcome to Aerius - Drone Building Survey API' });
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

module.exports = app;
