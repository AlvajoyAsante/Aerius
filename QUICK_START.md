# Aerius - Quick Start Guide

## Overview
Aerius is a full-stack application for managing drone-based building surveys. It provides a comprehensive solution for planning surveys, managing drone fleets, and maintaining building inspection records.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AlvajoyAsante/Aerius.git
cd Aerius
```

### 2. Backend Setup
```bash
cd backend
npm install
cp .env.example .env
```

Edit `.env` file:
```
PORT=5000
MONGODB_URI=mongodb://localhost:27017/aerius
```

Start the backend:
```bash
npm start
```

### 3. Frontend Setup
```bash
cd ../frontend
npm install
```

Optional - create `.env` file:
```
REACT_APP_API_URL=http://localhost:5000/api
```

Start the frontend:
```bash
npm start
```

## Usage

### Access the Application
Open your browser and navigate to: `http://localhost:3000`

### Main Features

1. **Home Page** - Overview and introduction to Aerius
2. **Surveys** - Create and manage drone building surveys
3. **Drones** - Manage your drone fleet inventory
4. **Buildings** - Maintain building database and inspection history

### Creating a Survey

1. Navigate to the **Surveys** page
2. Click **Create New Survey**
3. Fill in survey details:
   - Survey Name (required)
   - Location address and coordinates
   - Select drone and building
   - Set date and status
   - Add notes
4. Click **Create Survey**

### Adding a Drone

1. Navigate to the **Drones** page
2. Click **Add New Drone**
3. Fill in drone details:
   - Name and model (required)
   - Manufacturer and serial number
   - Specifications (altitude, flight time, camera, weight)
   - Status and maintenance dates
4. Click **Add Drone**

### Adding a Building

1. Navigate to the **Buildings** page
2. Click **Add New Building**
3. Fill in building details:
   - Building name (required)
   - Address and coordinates
   - Type and specifications
   - Owner information
4. Click **Add Building**

## MongoDB Setup

### Local MongoDB
Install MongoDB locally:
- macOS: `brew install mongodb-community`
- Ubuntu: `sudo apt-get install mongodb`
- Windows: Download from mongodb.com

Start MongoDB:
```bash
mongod
```

### MongoDB Atlas (Cloud)
1. Create account at mongodb.com/cloud/atlas
2. Create a cluster
3. Get connection string
4. Update `MONGODB_URI` in backend `.env` file

## Troubleshooting

### Backend won't start
- Ensure MongoDB is running
- Check `.env` file has correct MONGODB_URI
- Verify port 5000 is not in use

### Frontend won't connect to backend
- Ensure backend is running on port 5000
- Check `REACT_APP_API_URL` in frontend `.env`
- Clear browser cache

### Build errors
```bash
# Backend
cd backend
rm -rf node_modules package-lock.json
npm install

# Frontend
cd frontend
rm -rf node_modules package-lock.json
npm install
```

## API Documentation

### Surveys API
- `GET /api/surveys` - Get all surveys
- `GET /api/surveys/:id` - Get survey by ID
- `POST /api/surveys` - Create new survey
- `PUT /api/surveys/:id` - Update survey
- `DELETE /api/surveys/:id` - Delete survey

### Drones API
- `GET /api/drones` - Get all drones
- `GET /api/drones/:id` - Get drone by ID
- `POST /api/drones` - Create new drone
- `PUT /api/drones/:id` - Update drone
- `DELETE /api/drones/:id` - Delete drone

### Buildings API
- `GET /api/buildings` - Get all buildings
- `GET /api/buildings/:id` - Get building by ID
- `POST /api/buildings` - Create new building
- `PUT /api/buildings/:id` - Update building
- `DELETE /api/buildings/:id` - Delete building

## Development

### Running Tests
```bash
# Backend (when tests are added)
cd backend
npm test

# Frontend
cd frontend
npm test
```

### Building for Production
```bash
# Frontend
cd frontend
npm run build
```

The build folder can be deployed to any static hosting service.

## Support

For issues or questions:
1. Check the README.md file
2. Review the API documentation above
3. Open an issue on GitHub

## License
ISC License - See LICENSE file for details
