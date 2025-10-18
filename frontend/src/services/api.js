import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Survey API
export const surveyAPI = {
  getAll: () => api.get('/surveys'),
  getById: (id) => api.get(`/surveys/${id}`),
  create: (data) => api.post('/surveys', data),
  update: (id, data) => api.put(`/surveys/${id}`, data),
  delete: (id) => api.delete(`/surveys/${id}`),
};

// Drone API
export const droneAPI = {
  getAll: () => api.get('/drones'),
  getById: (id) => api.get(`/drones/${id}`),
  create: (data) => api.post('/drones', data),
  update: (id, data) => api.put(`/drones/${id}`, data),
  delete: (id) => api.delete(`/drones/${id}`),
};

// Building API
export const buildingAPI = {
  getAll: () => api.get('/buildings'),
  getById: (id) => api.get(`/buildings/${id}`),
  create: (data) => api.post('/buildings', data),
  update: (id, data) => api.put(`/buildings/${id}`, data),
  delete: (id) => api.delete(`/buildings/${id}`),
};

export default api;
