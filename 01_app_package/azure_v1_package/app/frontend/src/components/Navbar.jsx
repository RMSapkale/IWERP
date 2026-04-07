/**
 * Navbar.jsx — Top navigation bar for the IWERP Platform
 * Used on the Landing page (public) and within the app shell (authenticated).
 */
import { Link, useNavigate } from 'react-router-dom';
import { clearSession, getUser } from '../api';
import { Shield, LogOut, FlaskConical, LayoutDashboard, BarChart3 } from 'lucide-react';
import styles from './Navbar.module.css';

export default function Navbar({ authenticated = false }) {
  const navigate = useNavigate();
  const user = getUser();

  const handleLogout = () => {
    clearSession();
    navigate('/');
  };

  return (
    <nav className={styles.navbar}>
      <Link to="/" className={styles.logo}>
        <Shield size={22} color="#2563eb" />
        <span>IWERP</span>
      </Link>

      <div className={styles.links}>
        {!authenticated ? (
          <>
            <a href="#features" className={styles.navLink}>Platform</a>
            <a href="#products" className={styles.navLink}>Products</a>
            <a href="#capabilities" className={styles.navLink}>Capabilities</a>
            <Link to="/login" className={styles.navLink}>Sign In</Link>
            <Link to="/register" className="btn-primary" style={{ padding: '9px 20px', fontSize: '14px' }}>
              Get Access
            </Link>
          </>
        ) : (
          <>
            <Link to="/dashboard" className={styles.navLink}>
              <LayoutDashboard size={15} /> Dashboard
            </Link>
            <Link to="/lab" className={styles.navLink}>
              <FlaskConical size={15} /> Laboratory
            </Link>
            <Link to="/benchmarks" className={styles.navLink}>
              <BarChart3 size={15} /> Benchmarks
            </Link>
            <div className={styles.userPill}>
              <div className={styles.dot} />
              {user}
            </div>
            <button className={styles.logoutBtn} onClick={handleLogout}>
              <LogOut size={15} /> Sign Out
            </button>
          </>
        )}
      </div>
    </nav>
  );
}
