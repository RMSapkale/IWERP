import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Landing from './pages/Landing';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import Laboratory from './pages/Laboratory';
import Benchmarks from './pages/Benchmarks';
import Settings from './pages/Settings';
import './index.css';

function PrivateRoute({ children }) {
  const loggedIn = localStorage.getItem('iwerp_token') || localStorage.getItem('iwerp_key');
  return loggedIn ? children : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/dashboard" element={<PrivateRoute><Dashboard /></PrivateRoute>} />
        <Route path="/lab" element={<PrivateRoute><Laboratory /></PrivateRoute>} />
        <Route path="/benchmarks" element={<PrivateRoute><Benchmarks /></PrivateRoute>} />
        <Route path="/settings" element={<PrivateRoute><Settings /></PrivateRoute>} />
      </Routes>
    </BrowserRouter>
  );
}
