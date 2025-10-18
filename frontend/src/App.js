import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import Home from './pages/Home';
import Surveys from './pages/Surveys';
import Drones from './pages/Drones';
import Buildings from './pages/Buildings';

function App() {
  return (
    <Router>
      <div className="App">
        <nav className="navbar">
          <div className="nav-brand">
            <h1>Aerius</h1>
            <p>Drone Building Survey System</p>
          </div>
          <ul className="nav-links">
            <li><Link to="/">Home</Link></li>
            <li><Link to="/surveys">Surveys</Link></li>
            <li><Link to="/drones">Drones</Link></li>
            <li><Link to="/buildings">Buildings</Link></li>
          </ul>
        </nav>
        
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/surveys" element={<Surveys />} />
            <Route path="/drones" element={<Drones />} />
            <Route path="/buildings" element={<Buildings />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
