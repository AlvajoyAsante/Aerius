import React, { useState, useEffect } from 'react';
import { surveyAPI, droneAPI, buildingAPI } from '../services/api';

function Surveys() {
  const [surveys, setSurveys] = useState([]);
  const [drones, setDrones] = useState([]);
  const [buildings, setBuildings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    location: { address: '', coordinates: { lat: '', lng: '' } },
    droneId: '',
    buildingId: '',
    date: new Date().toISOString().split('T')[0],
    status: 'planned',
    notes: '',
  });

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [surveysRes, dronesRes, buildingsRes] = await Promise.all([
        surveyAPI.getAll(),
        droneAPI.getAll(),
        buildingAPI.getAll(),
      ]);
      setSurveys(surveysRes.data);
      setDrones(dronesRes.data);
      setBuildings(buildingsRes.data);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to fetch data');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await surveyAPI.create(formData);
      setShowModal(false);
      fetchData();
      setFormData({
        name: '',
        location: { address: '', coordinates: { lat: '', lng: '' } },
        droneId: '',
        buildingId: '',
        date: new Date().toISOString().split('T')[0],
        status: 'planned',
        notes: '',
      });
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to create survey');
    }
  };

  const handleDelete = async (id) => {
    if (window.confirm('Are you sure you want to delete this survey?')) {
      try {
        await surveyAPI.delete(id);
        fetchData();
      } catch (err) {
        setError(err.response?.data?.message || 'Failed to delete survey');
      }
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    if (name.includes('.')) {
      const [parent, child, grandchild] = name.split('.');
      if (grandchild) {
        setFormData(prev => ({
          ...prev,
          [parent]: {
            ...prev[parent],
            [child]: {
              ...prev[parent][child],
              [grandchild]: value
            }
          }
        }));
      } else {
        setFormData(prev => ({
          ...prev,
          [parent]: {
            ...prev[parent],
            [child]: value
          }
        }));
      }
    } else {
      setFormData(prev => ({ ...prev, [name]: value }));
    }
  };

  if (loading) return <div className="loading">Loading surveys...</div>;

  return (
    <div>
      <div className="page-header">
        <h2>Surveys</h2>
        <p>Manage your building survey operations</p>
      </div>

      {error && <div className="error">{error}</div>}

      <button className="btn btn-primary" onClick={() => setShowModal(true)}>
        Create New Survey
      </button>

      <div style={{ marginTop: '2rem' }}>
        {surveys.length === 0 ? (
          <div className="card">
            <p>No surveys found. Create your first survey to get started.</p>
          </div>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Location</th>
                <th>Date</th>
                <th>Status</th>
                <th>Drone</th>
                <th>Building</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {surveys.map((survey) => (
                <tr key={survey._id}>
                  <td>{survey.name}</td>
                  <td>{survey.location?.address || 'N/A'}</td>
                  <td>{new Date(survey.date).toLocaleDateString()}</td>
                  <td>
                    <span className={`status-badge status-${survey.status}`}>
                      {survey.status}
                    </span>
                  </td>
                  <td>{survey.droneId?.name || 'N/A'}</td>
                  <td>{survey.buildingId?.name || 'N/A'}</td>
                  <td>
                    <button
                      className="btn btn-danger"
                      onClick={() => handleDelete(survey._id)}
                      style={{ padding: '0.5rem 1rem' }}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {showModal && (
        <div className="modal-overlay">
          <div className="modal">
            <div className="modal-header">
              <h3>Create New Survey</h3>
              <button className="close-btn" onClick={() => setShowModal(false)}>
                &times;
              </button>
            </div>
            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label>Survey Name *</label>
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label>Address</label>
                <input
                  type="text"
                  name="location.address"
                  value={formData.location.address}
                  onChange={handleChange}
                />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="form-group">
                  <label>Latitude</label>
                  <input
                    type="number"
                    step="any"
                    name="location.coordinates.lat"
                    value={formData.location.coordinates.lat}
                    onChange={handleChange}
                  />
                </div>

                <div className="form-group">
                  <label>Longitude</label>
                  <input
                    type="number"
                    step="any"
                    name="location.coordinates.lng"
                    value={formData.location.coordinates.lng}
                    onChange={handleChange}
                  />
                </div>
              </div>

              <div className="form-group">
                <label>Drone</label>
                <select name="droneId" value={formData.droneId} onChange={handleChange}>
                  <option value="">Select a drone</option>
                  {drones.map((drone) => (
                    <option key={drone._id} value={drone._id}>
                      {drone.name} - {drone.model}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label>Building</label>
                <select name="buildingId" value={formData.buildingId} onChange={handleChange}>
                  <option value="">Select a building</option>
                  {buildings.map((building) => (
                    <option key={building._id} value={building._id}>
                      {building.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label>Date *</label>
                <input
                  type="date"
                  name="date"
                  value={formData.date}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label>Status</label>
                <select name="status" value={formData.status} onChange={handleChange}>
                  <option value="planned">Planned</option>
                  <option value="in-progress">In Progress</option>
                  <option value="completed">Completed</option>
                  <option value="cancelled">Cancelled</option>
                </select>
              </div>

              <div className="form-group">
                <label>Notes</label>
                <textarea
                  name="notes"
                  value={formData.notes}
                  onChange={handleChange}
                />
              </div>

              <div className="action-buttons">
                <button type="submit" className="btn btn-primary">
                  Create Survey
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

export default Surveys;
