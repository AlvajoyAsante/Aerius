# Aerius

A full-stack application for drone-building surveying. Aerius is a comprehensive management system designed to streamline drone-based building inspections and surveys.

## Features

- **Survey Management**: Plan, execute, and track drone surveys for building inspections
- **Drone Fleet Management**: Maintain inventory, specifications, and maintenance schedules
- **Building Database**: Store comprehensive building information and inspection history
- **Real-time Status Tracking**: Monitor survey progress and drone availability
- **RESTful API**: Complete backend API for all operations

## Tech Stack

### Backend
- Node.js with Express.js
- MongoDB with Mongoose ODM
- RESTful API architecture
- CORS enabled for cross-origin requests

### Frontend
- React.js
- React Router for navigation
- Axios for API calls
- Modern, responsive UI design

## Prerequisites

- Node.js (v14 or higher)
- MongoDB (local or cloud instance)
- npm or yarn package manager

## Installation

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

4. Configure your environment variables in `.env`:
```
PORT=5000
MONGODB_URI=mongodb://localhost:27017/aerius
```

5. Start the backend server:
```bash
npm start
```

For development with auto-reload:
```bash
npm run dev
```

The backend will run on `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file (optional):
```
REACT_APP_API_URL=http://localhost:5000/api
```

4. Start the frontend development server:
```bash
npm start
```

The frontend will run on `http://localhost:3000`

## Project Structure

```
Aerius/
├── backend/
│   ├── models/           # MongoDB models
│   │   ├── Survey.js
│   │   ├── Drone.js
│   │   └── Building.js
│   ├── routes/           # API routes
│   │   ├── surveys.js
│   │   ├── drones.js
│   │   └── buildings.js
│   ├── controllers/      # Request handlers
│   │   ├── surveyController.js
│   │   ├── droneController.js
│   │   └── buildingController.js
│   ├── server.js         # Express server setup
│   └── package.json
│
└── frontend/
    ├── src/
    │   ├── components/   # Reusable components
    │   ├── pages/        # Page components
    │   │   ├── Home.js
    │   │   ├── Surveys.js
    │   │   ├── Drones.js
    │   │   └── Buildings.js
    │   ├── services/     # API services
    │   │   └── api.js
    │   ├── App.js
    │   └── App.css
    └── package.json
```

## API Endpoints

### Surveys
- `GET /api/surveys` - Get all surveys
- `GET /api/surveys/:id` - Get single survey
- `POST /api/surveys` - Create new survey
- `PUT /api/surveys/:id` - Update survey
- `DELETE /api/surveys/:id` - Delete survey

### Drones
- `GET /api/drones` - Get all drones
- `GET /api/drones/:id` - Get single drone
- `POST /api/drones` - Create new drone
- `PUT /api/drones/:id` - Update drone
- `DELETE /api/drones/:id` - Delete drone

### Buildings
- `GET /api/buildings` - Get all buildings
- `GET /api/buildings/:id` - Get single building
- `POST /api/buildings` - Create new building
- `PUT /api/buildings/:id` - Update building
- `DELETE /api/buildings/:id` - Delete building

## Usage

1. Start both backend and frontend servers
2. Access the application at `http://localhost:3000`
3. Navigate through different sections:
   - **Home**: Overview and introduction
   - **Surveys**: Create and manage building surveys
   - **Drones**: Manage your drone fleet
   - **Buildings**: Maintain building database

## Data Models

### Survey
- Name, location, coordinates
- Associated drone and building
- Date and status (planned, in-progress, completed, cancelled)
- Survey data (altitude, weather, etc.)
- Notes and images

### Drone
- Name, model, manufacturer
- Serial number and specifications
- Status (available, in-use, maintenance, retired)
- Maintenance and purchase dates

### Building
- Name and address
- Coordinates and type
- Specifications (height, floors, area, etc.)
- Owner information
- Inspection history

## Development

### Backend Development
```bash
cd backend
npm run dev
```

### Frontend Development
```bash
cd frontend
npm start
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the ISC License - see the LICENSE file for details.

## Author

Aerius Development Team

## Acknowledgments

- Built with modern web technologies
- Designed for professional drone surveying operations
- Scalable architecture for future enhancements