import React, { useState, useEffect } from 'react';
import { buildingAPI } from '../services/api';

function Buildings() {
  const [buildings, setBuildings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    address: {
      street: '',
      city: '',
      state: '',
      zipCode: '',
      country: '',
    },
    coordinates: {
      lat: '',
      lng: '',
    },
    type: 'residential',
    specifications: {
      height: '',
      floors: '',
      constructionYear: '',
      roofType: '',
      area: '',
    },
    owner: {
      name: '',
      contact: '',
      email: '',
    },
  });

  useEffect(() => {
    fetchBuildings();
  }, []);

  const fetchBuildings = async () => {
    try {
      setLoading(true);
      const response = await buildingAPI.getAll();
      setBuildings(response.data);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to fetch buildings');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await buildingAPI.create(formData);
      setShowModal(false);
      fetchBuildings();
      setFormData({
        name: '',
        address: {
          street: '',
          city: '',
          state: '',
          zipCode: '',
          country: '',
        },
        coordinates: {
          lat: '',
          lng: '',
        },
        type: 'residential',
        specifications: {
          height: '',
          floors: '',
          constructionYear: '',
          roofType: '',
          area: '',
        },
        owner: {
          name: '',
          contact: '',
          email: '',
        },
      });
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to create building');
    }
  };

  const handleDelete = async (id) => {
    if (window.confirm('Are you sure you want to delete this building?')) {
      try {
        await buildingAPI.delete(id);
        fetchBuildings();
      } catch (err) {
        setError(err.response?.data?.message || 'Failed to delete building');
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

  if (loading) return <div className="loading">Loading buildings...</div>;

  return (
    <div>
      <div className="page-header">
        <h2>Buildings</h2>
        <p>Manage your building database and inspection records</p>
      </div>

      {error && <div className="error">{error}</div>}

      <button className="btn btn-primary" onClick={() => setShowModal(true)}>
        Add New Building
      </button>

      <div className="grid" style={{ marginTop: '2rem' }}>
        {buildings.length === 0 ? (
          <div className="card">
            <p>No buildings found. Add your first building to get started.</p>
          </div>
        ) : (
          buildings.map((building) => (
            <div key={building._id} className="card">
              <h3>{building.name}</h3>
              
              {building.address && (
                <div style={{ marginTop: '0.5rem' }}>
                  <p><strong>Address:</strong></p>
                  <p style={{ marginLeft: '1rem' }}>
                    {building.address.street && `${building.address.street}, `}
                    {building.address.city && `${building.address.city}, `}
                    {building.address.state && `${building.address.state} `}
                    {building.address.zipCode}
                  </p>
                </div>
              )}

              {building.type && (
                <p style={{ marginTop: '0.5rem' }}>
                  <strong>Type:</strong> {building.type.charAt(0).toUpperCase() + building.type.slice(1)}
                </p>
              )}

              {building.specifications && (
                <div style={{ marginTop: '1rem' }}>
                  <h4>Specifications:</h4>
                  <ul style={{ marginLeft: '1.5rem', marginTop: '0.5rem' }}>
                    {building.specifications.floors && (
                      <li>Floors: {building.specifications.floors}</li>
                    )}
                    {building.specifications.height && (
                      <li>Height: {building.specifications.height}m</li>
                    )}
                    {building.specifications.area && (
                      <li>Area: {building.specifications.area}m²</li>
                    )}
                    {building.specifications.constructionYear && (
                      <li>Built: {building.specifications.constructionYear}</li>
                    )}
                    {building.specifications.roofType && (
                      <li>Roof Type: {building.specifications.roofType}</li>
                    )}
                  </ul>
                </div>
              )}

              {building.owner && building.owner.name && (
                <div style={{ marginTop: '1rem' }}>
                  <p><strong>Owner:</strong> {building.owner.name}</p>
                  {building.owner.contact && <p><strong>Contact:</strong> {building.owner.contact}</p>}
                  {building.owner.email && <p><strong>Email:</strong> {building.owner.email}</p>}
                </div>
              )}

              <button
                className="btn btn-danger"
                onClick={() => handleDelete(building._id)}
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
              <h3>Add New Building</h3>
              <button className="close-btn" onClick={() => setShowModal(false)}>
                &times;
              </button>
            </div>
            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label>Building Name *</label>
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  required
                />
              </div>

              <h4>Address</h4>

              <div className="form-group">
                <label>Street</label>
                <input
                  type="text"
                  name="address.street"
                  value={formData.address.street}
                  onChange={handleChange}
                />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="form-group">
                  <label>City</label>
                  <input
                    type="text"
                    name="address.city"
                    value={formData.address.city}
                    onChange={handleChange}
                  />
                </div>

                <div className="form-group">
                  <label>State</label>
                  <input
                    type="text"
                    name="address.state"
                    value={formData.address.state}
                    onChange={handleChange}
                  />
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="form-group">
                  <label>Zip Code</label>
                  <input
                    type="text"
                    name="address.zipCode"
                    value={formData.address.zipCode}
                    onChange={handleChange}
                  />
                </div>

                <div className="form-group">
                  <label>Country</label>
                  <input
                    type="text"
                    name="address.country"
                    value={formData.address.country}
                    onChange={handleChange}
                  />
                </div>
              </div>

              <h4>Coordinates</h4>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="form-group">
                  <label>Latitude</label>
                  <input
                    type="number"
                    step="any"
                    name="coordinates.lat"
                    value={formData.coordinates.lat}
                    onChange={handleChange}
                  />
                </div>

                <div className="form-group">
                  <label>Longitude</label>
                  <input
                    type="number"
                    step="any"
                    name="coordinates.lng"
                    value={formData.coordinates.lng}
                    onChange={handleChange}
                  />
                </div>
              </div>

              <div className="form-group">
                <label>Building Type</label>
                <select name="type" value={formData.type} onChange={handleChange}>
                  <option value="residential">Residential</option>
                  <option value="commercial">Commercial</option>
                  <option value="industrial">Industrial</option>
                  <option value="institutional">Institutional</option>
                  <option value="other">Other</option>
                </select>
              </div>

              <h4>Specifications</h4>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="form-group">
                  <label>Height (m)</label>
                  <input
                    type="number"
                    name="specifications.height"
                    value={formData.specifications.height}
                    onChange={handleChange}
                  />
                </div>

                <div className="form-group">
                  <label>Number of Floors</label>
                  <input
                    type="number"
                    name="specifications.floors"
                    value={formData.specifications.floors}
                    onChange={handleChange}
                  />
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="form-group">
                  <label>Construction Year</label>
                  <input
                    type="number"
                    name="specifications.constructionYear"
                    value={formData.specifications.constructionYear}
                    onChange={handleChange}
                  />
                </div>

                <div className="form-group">
                  <label>Area (m²)</label>
                  <input
                    type="number"
                    name="specifications.area"
                    value={formData.specifications.area}
                    onChange={handleChange}
                  />
                </div>
              </div>

              <div className="form-group">
                <label>Roof Type</label>
                <input
                  type="text"
                  name="specifications.roofType"
                  value={formData.specifications.roofType}
                  onChange={handleChange}
                  placeholder="e.g., Flat, Gable, Hip"
                />
              </div>

              <h4>Owner Information</h4>

              <div className="form-group">
                <label>Owner Name</label>
                <input
                  type="text"
                  name="owner.name"
                  value={formData.owner.name}
                  onChange={handleChange}
                />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="form-group">
                  <label>Contact</label>
                  <input
                    type="text"
                    name="owner.contact"
                    value={formData.owner.contact}
                    onChange={handleChange}
                  />
                </div>

                <div className="form-group">
                  <label>Email</label>
                  <input
                    type="email"
                    name="owner.email"
                    value={formData.owner.email}
                    onChange={handleChange}
                  />
                </div>
              </div>

              <div className="action-buttons">
                <button type="submit" className="btn btn-primary">
                  Add Building
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

export default Buildings;
