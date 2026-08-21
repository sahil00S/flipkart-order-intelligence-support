import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

export default function Dashboard() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  return (
    <div>
      {/* Hero banner */}
      <div
        className="card"
        style={{
          background:
            "linear-gradient(135deg, #1a56db 0%, #0f3a94 100%)",
          color: "#fff",
          marginBottom: 24,
          padding: "32px 36px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 20 }}>
          <div>
            <h2 style={{ fontSize: "1.35rem", fontWeight: 700, marginBottom: 6 }}>
              Flipkart Order Intelligence & Support
            </h2>
            <p style={{ fontSize: "0.9rem", opacity: 0.88, maxWidth: 600 }}>
              An end-to-end AI system combining machine learning, computer vision,
              retrieval-augmented generation, and a LangGraph support assistant
              into one connected workflow.
            </p>
          </div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <Link to="/return-risk" className="btn btn-sm" style={{ background: "#fff", color: "#1a56db" }}>
              Return Risk
            </Link>
            <Link to="/policy-assistant" className="btn btn-sm" style={{ background: "#fff", color: "#1a56db" }}>
              Policy Assistant
            </Link>
            <Link to="/image-classifier" className="btn btn-sm" style={{ background: "#fff", color: "#1a56db" }}>
              Image Classifier
            </Link>
          </div>
        </div>
      </div>

      {/* System status */}
      <div className="stats-grid" style={{ marginBottom: 24 }}>
        <div className="stat-card">
          <div className="stat-label">System Status</div>
          <div className="stat-value" style={{ color: health ? "var(--color-success)" : "var(--color-danger)", fontSize: "1.1rem" }}>
            {health ? "● Online" : "● Checking..."}
          </div>
          <div className="stat-detail">API: /api/health</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Dataset</div>
          <div className="stat-value">6,000</div>
          <div className="stat-detail">Synthetic orders (seed 42)</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Knowledge Base</div>
          <div className="stat-value">12</div>
          <div className="stat-detail">Policy documents · 37 chunks</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Sample Images</div>
          <div className="stat-value">5</div>
          <div className="stat-detail">Fashion-MNIST test images</div>
        </div>
      </div>

      {/* Capability cards */}
      <h3 style={{ fontSize: "0.95rem", fontWeight: 600, marginBottom: 14 }}>
        System Capabilities
      </h3>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16, marginBottom: 24 }}>
        <CapabilityCard
          icon="📈"
          title="Return Risk Prediction"
          desc="Random Forest classifier tuned via GridSearchCV on 5-fold StratifiedKFold cross-validation."
          metrics={[
            { label: "Model", value: "RandomForest" },
            { label: "Test ROC-AUC", value: "0.6203" },
            { label: "CV ROC-AUC", value: "0.6192" },
            { label: "Threshold (t*)", value: "0.50" },
          ]}
          link="/return-risk"
          linkLabel="Try It →"
        />
        <CapabilityCard
          icon="💬"
          title="Policy Assistant"
          desc="LangGraph agent with FAISS RAG, grounded answers, prompt-injection protection, and conversation state."
          metrics={[
            { label: "Architecture", value: "LangGraph" },
            { label: "Embeddings", value: "MiniLM-L6" },
            { label: "Precision@3", value: "0.7222" },
            { label: "Recall@3", value: "1.0000" },
          ]}
          link="/policy-assistant"
          linkLabel="Try It →"
        />
        <CapabilityCard
          icon="🖼️"
          title="Product Image Classifier"
          desc="Transfer-learning ResNet-18 fine-tuned on Fashion-MNIST for 10-class product classification."
          metrics={[
            { label: "Architecture", value: "ResNet-18" },
            { label: "Test Accuracy", value: "88.49%" },
            { label: "Val Accuracy", value: "89.64%" },
            { label: "Classes", value: "10" },
          ]}
          link="/image-classifier"
          linkLabel="Try It →"
        />
      </div>

      {/* Tech stack */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">Technology Stack</div>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {[
            "scikit-learn",
            "Random Forest",
            "PyTorch",
            "ResNet-18",
            "Fashion-MNIST",
            "sentence-transformers",
            "FAISS",
            "LangGraph",
            "MOCK_LLM",
            "Prompt Guardrails",
            "Groundedness Check",
            "FastAPI",
            "React",
            "Vite",
          ].map((tech) => (
            <span
              key={tech}
              style={{
                padding: "5px 12px",
                borderRadius: 20,
                fontSize: "0.78rem",
                fontWeight: 500,
                background: "var(--color-primary-light)",
                color: "var(--color-primary)",
              }}
            >
              {tech}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

function CapabilityCard({ icon, title, desc, metrics, link, linkLabel }) {
  return (
    <div className="card" style={{ display: "flex", flexDirection: "column" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
        <span style={{ fontSize: "1.4rem" }}>{icon}</span>
        <div>
          <div className="card-title">{title}</div>
        </div>
      </div>
      <p style={{ fontSize: "0.82rem", color: "var(--color-text-secondary)", marginBottom: 14, lineHeight: 1.5 }}>
        {desc}
      </p>
      <div style={{ flex: 1 }}>
        {metrics.map((m) => (
          <div
            key={m.label}
            style={{
              display: "flex",
              justifyContent: "space-between",
              padding: "5px 0",
              borderBottom: "1px solid var(--color-border-light)",
              fontSize: "0.8rem",
            }}
          >
            <span style={{ color: "var(--color-text-muted)" }}>{m.label}</span>
            <span style={{ fontWeight: 600, color: "var(--color-text)" }}>{m.value}</span>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 14 }}>
        <Link to={link} className="btn btn-secondary btn-sm" style={{ width: "100%" }}>
          {linkLabel}
        </Link>
      </div>
    </div>
  );
}
