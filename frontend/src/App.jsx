import { Routes, Route, useLocation } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import ReturnRisk from "./pages/ReturnRisk";
import PolicyAssistant from "./pages/PolicyAssistant";
import ImageClassifier from "./pages/ImageClassifier";
import ModelInsights from "./pages/ModelInsights";

const pageTitles = {
  "/": "Dashboard",
  "/return-risk": "Return Risk Prediction",
  "/policy-assistant": "Policy Assistant",
  "/image-classifier": "Product Image Classifier",
  "/model-insights": "Model Insights",
};

export default function App() {
  const location = useLocation();
  const title = pageTitles[location.pathname] || "Dashboard";

  return (
    <div className="app-layout">
      <Sidebar />
      <main className="app-main">
        <header className="app-header">
          <h1>{title}</h1>
          <span className="header-badge">AI/ML Capstone</span>
        </header>
        <div className="app-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/return-risk" element={<ReturnRisk />} />
            <Route path="/policy-assistant" element={<PolicyAssistant />} />
            <Route path="/image-classifier" element={<ImageClassifier />} />
            <Route path="/model-insights" element={<ModelInsights />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}
