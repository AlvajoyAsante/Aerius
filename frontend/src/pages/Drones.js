import React, { useState, useEffect } from 'react';
import { droneAPI } from '../services/api';

function Drones() {
  const [drones, setDrones] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    model: '',
    manufacturer: '',
    serialNumber: '',
    specifications: {
      maxAltitude: '',
      maxFlightTime: '',
      cameraResolution: '',
      weight: '',
    },
    status: 'available',
    lastMaintenance: '',
    purchaseDate: '',
  });

  useEffect(() => {
    fetchDrones();
  }, []);

  const fetchDrones = async () => {
    try {
      setLoading(true);
      const response = await droneAPI.getAll();
      setDrones(response.data);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to fetch drones');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await droneAPI.create(formData);
      setShowModal(false);
      fetchDrones();
      setFormData({
        name: '',
        model: '',
        manufacturer: '',
        serialNumber: '',
        specifications: {
          maxAltitude: '',
          maxFlightTime: '',
          cameraResolution: '',
          weight: '',
        },
        status: 'available',
        lastMaintenance: '',
        purchaseDate: '',
      });
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to create drone');
    }
  };

  const handleDelete = async (id) => {
    if (window.confirm('Are you sure you want to delete this drone?')) {
      try {
        await droneAPI.delete(id);
        fetchDrones();
      } catch (err) {
        setError(err.response?.data?.message || 'Failed to delete drone');
      }
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    if (name.includes('.')) {
      const [parent, child] = name.split('.');
      setFormData(prev => ({
        ...prev,
        [parent]: {
          ...prev[parent],
          [child]: value
        }
      }));
    } else {
      setFormData(prev => ({ ...prev, [name]: value }));
    }
  };

  if (loading) return <div className="loading">Loading drones...</div>;

  return (
    <div>
      <div className="page-header">
        <h2>Drone Fleet</h2>
        <p>Manage your drone inventory and specifications</p>
      </div>

      {error && <div className="error">{error}</div>}

      <button className="btn btn-primary" onClick={() => setShowModal(true)}>
        Add New Drone
      </button>

      <div className="grid" style={{ marginTop: '2rem' }}>
        {drones.length === 0 ? (
          <div className="card">
            <p>No drones found. Add your first drone to get started.</p>
          </div>
        ) : (
          drones.map((drone) => (
            <div key={drone._id} className="card">
              <h3>{drone.name}</h3>
              <p><strong>Model:</strong> {drone.model}</p>
              {drone.manufacturer && <p><strong>Manufacturer:</strong> {drone.manufacturer}</p>}
              {drone.serialNumber && <p><strong>Serial Number:</strong> {drone.serialNumber}</p>}
              
              <div style={{ marginTop: '1rem', marginBottom: '1rem' }}>
                <span className={`status-badge status-${drone.status}`}>
                  {drone.status}
                </span>
              </div>

              {drone.specifications && (
                <div style={{ marginTop: '1rem' }}>
                  <h4>Specifications:</h4>
                  <ul style={{ marginLeft: '1.5rem', marginTop: '0.5rem' }}>
                    {drone.specifications.maxAltitude && (
                      <li>Max Altitude: {drone.specifications.maxAltitude}m</li>
                    )}
                    {drone.specifications.maxFlightTime && (
                      <li>Flight Time: {drone.specifications.maxFlightTime} min</li>
                    )}
                    {drone.specifications.cameraResolution && (
                      <li>Camera: {drone.specifications.cameraResolution}</li>
                    )}
                    {drone.specifications.weight && (
                      <li>Weight: {drone.specifications.weight}kg</li>
                    )}
                  </ul>
                </div>
              )}

              <button
                className="btn btn-danger"
                onClick={() => handleDelete(drone._id)}
                style={{ marginTop: '1rem' }}
              >
                Delete
              </button>
            </div>
          ))
        )}
      </div>

      {showModal && (
        <div className="modal-overlay">
          <div className="modal">
            <div className="modal-header">
              <h3>Add New Drone</h3>
              <button className="close-btn" onClick={() => setShowModal(false)}>
                &times;
              </button>
            </div>
            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label>Drone Name *</label>
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label>Model *</label>
                <input
                  type="text"
                  name="model"
                  value={formData.model}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label>Manufacturer</label>
                <input
                  type="text"
                  name="manufacturer"
                  value={formData.manufacturer}
                  onChange={handleChange}
                />
              </div>

              <div className="form-group">
                <label>Serial Number</label>
                <input
                  type="text"
                  name="serialNumber"
                  value={formData.serialNumber}
                  onChange={handleChange}
                />
              </div>

              <h4>Specifications</h4>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="form-group">
                  <label>Max Altitude (m)</label>
                  <input
                    type="number"
                    name="specifications.maxAltitude"
                    value={formData.specifications.maxAltitude}
                    onChange={handleChange}
                  />
                </div>

                <div className="form-group">
                  <label>Max Flight Time (min)</label>
                  <input
                    type="number"
                    name="specifications.maxFlightTime"
                    value={formData.specifications.maxFlightTime}
                    onChange={handleChange}
                  />
                </div>
              </div>

              <div className="form-group">
                <label>Camera Resolution</label>
                <input
                  type="text"
                  name="specifications.cameraResolution"
                  value={formData.specifications.cameraResolution}
                  onChange={handleChange}
                  placeholder="e.g., 4K, 12MP"
                />
              </div>

              <div className="form-group">
                <label>Weight (kg)</label>
                <input
                  type="number"
                  step="0.01"
                  name="specifications.weight"
                  value={formData.specifications.weight}
                  onChange={handleChange}
                />
              </div>

              <div className="form-group">
                <label>Status</label>
                <select name="status" value={formData.status} onChange={handleChange}>
                  <option value="available">Available</option>
                  <option value="in-use">In Use</option>
                  <option value="maintenance">Maintenance</option>
                  <option value="retired">Retired</option>
                </select>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="form-group">
                  <label>Last Maintenance</label>
                  <input
                    type="date"
                    name="lastMaintenance"
                    value={formData.lastMaintenance}
                    onChange={handleChange}
                  />
                </div>

                <div className="form-group">
                  <label>Purchase Date</label>
                  <input
                    type="date"
                    name="purchaseDate"
                    value={formData.purchaseDate}
                    onChange={handleChange}
                  />
                </div>
              </div>

              <div className="action-buttons">
                <button type="submit" className="btn btn-primary">
                  Add Drone
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => setShowModal(false)}
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

export default Drones;
