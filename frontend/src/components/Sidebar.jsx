import { NavLink } from "react-router-dom";

const navItems = [
  {
    to: "/",
    label: "Dashboard",
    icon: "📊",
  },
  {
    to: "/return-risk",
    label: "Return Risk",
    icon: "⚠️",
  },
  {
    to: "/policy-assistant",
    label: "Policy Assistant",
    icon: "💬",
  },
  {
    to: "/image-classifier",
    label: "Image Classifier",
    icon: "🖼️",
  },
  {
    to: "/model-insights",
    label: "Model Insights",
    icon: "🔬",
  },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <h2>Flipkart Order Intelligence</h2>
        <p>AI-Powered Support Dashboard</p>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) =>
              isActive ? "active" : undefined
            }
          >
            <span className="nav-icon">{item.icon}</span>
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <span>
          <span className="status-dot" />
          System Online
        </span>
      </div>
    </aside>
  );
}
