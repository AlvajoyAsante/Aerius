import React from 'react';

function Home() {
  return (
    <div className="page-header">
      <h2>Welcome to Aerius</h2>
      <p>Professional Drone Building Survey System</p>
      
      <div className="grid" style={{ marginTop: '2rem' }}>
        <div className="card">
          <h3>📋 Survey Management</h3>
          <p>Plan, execute, and track drone surveys for building inspections.</p>
        </div>
        
        <div className="card">
          <h3>🚁 Drone Fleet</h3>
          <p>Manage your drone inventory, maintenance schedules, and specifications.</p>
        </div>
        
        <div className="card">
          <h3>🏢 Building Database</h3>
          <p>Maintain comprehensive records of buildings and inspection history.</p>
        </div>
      </div>
      
      <div className="card" style={{ marginTop: '2rem' }}>
        <h3>About Aerius</h3>
        <p>
          Aerius is a comprehensive full-stack application designed for managing drone-based 
          building surveys. It provides tools for planning surveys, tracking drone operations, 
          and maintaining detailed building inspection records.
        </p>
        <h4 style={{ marginTop: '1.5rem' }}>Key Features:</h4>
        <ul style={{ marginLeft: '2rem', marginTop: '0.5rem' }}>
          <li>Create and manage building surveys</li>
          <li>Track drone fleet and availability</li>
          <li>Maintain building inspection history</li>
          <li>Monitor survey status and progress</li>
          <li>Store survey data and images</li>
          <li>Generate inspection reports</li>
        </ul>
      </div>
    </div>
  );
}

export default Home;
