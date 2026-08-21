const PER_CLASS_METRICS = [
  { class_id: 0, class_name: "T-shirt_top", precision: 0.8421, recall: 0.832, f1: 0.837 },
  { class_id: 1, class_name: "Trouser", precision: 0.9888, recall: 0.971, f1: 0.9798 },
  { class_id: 2, class_name: "Pullover", precision: 0.8455, recall: 0.859, f1: 0.8522 },
  { class_id: 3, class_name: "Dress", precision: 0.8697, recall: 0.874, f1: 0.8718 },
  { class_id: 4, class_name: "Coat", precision: 0.8166, recall: 0.797, f1: 0.8067 },
  { class_id: 5, class_name: "Sandal", precision: 0.9731, recall: 0.941, f1: 0.9568 },
  { class_id: 6, class_name: "Shirt", precision: 0.6693, recall: 0.688, f1: 0.6785 },
  { class_id: 7, class_name: "Sneaker", precision: 0.9205, recall: 0.961, f1: 0.9403 },
  { class_id: 8, class_name: "Bag", precision: 0.9741, recall: 0.979, f1: 0.9766 },
  { class_id: 9, class_name: "Ankle_boot", precision: 0.9575, recall: 0.947, f1: 0.9522 },
];

const RETRIEVAL_METRICS = [
  { query: "How many days can an apparel product be returned?", relevant: "return_windows", p3: 1.0, r3: 1.0 },
  { query: "How long does a COD refund take?", relevant: "cod_refund", p3: 0.6667, r3: 1.0 },
  { query: "What is the standard delivery time?", relevant: "delivery_sla", p3: 0.6667, r3: 1.0 },
  { query: "When is reverse pickup available?", relevant: "reverse_pickup", p3: 0.6667, r3: 1.0 },
  { query: "Can a customer request a replacement?", relevant: "replacement_policy", p3: 0.6667, r3: 1.0 },
  { query: "What happens for a damaged product?", relevant: "damaged_product", p3: 0.6667, r3: 1.0 },
];

const CONFUSION_MATRIX = [
  [832, 1, 20, 13, 15, 0, 108, 0, 9, 2],
  [0, 971, 1, 17, 4, 0, 2, 0, 5, 0],
  [14, 0, 859, 14, 65, 0, 45, 0, 3, 0],
  [16, 12, 14, 874, 55, 0, 20, 0, 9, 0],
  [0, 1, 53, 38, 797, 0, 105, 0, 6, 0],
  [0, 0, 1, 0, 0, 941, 0, 23, 0, 35],
  [90, 1, 53, 24, 90, 0, 688, 0, 50, 4],
  [0, 0, 0, 0, 0, 14, 0, 961, 0, 25],
  [1, 0, 3, 6, 3, 0, 4, 0, 979, 4],
  [0, 0, 0, 0, 0, 17, 0, 31, 2, 950],
];

const CLASS_NAMES = [
  "T-shirt_top", "Trouser", "Pullover", "Dress", "Coat",
  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot",
];

export default function ModelInsights() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {/* Overview stats */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-label">Return-Risk Model</div>
          <div className="stat-value">Random Forest</div>
          <div className="stat-detail">200 trees · max_depth=6</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Test ROC-AUC</div>
          <div className="stat-value">0.6203</div>
          <div className="stat-detail">CV: 0.6192</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Image Classifier</div>
          <div className="stat-value">88.49%</div>
          <div className="stat-detail">ResNet-18 · Fashion-MNIST</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">RAG Precision@3</div>
          <div className="stat-value">0.7222</div>
          <div className="stat-detail">Recall@3: 1.0000</div>
        </div>
      </div>

      {/* Two-column layout */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, alignItems: "start" }}>
        {/* Return Risk Model */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">Return-Risk Model</div>
          </div>

          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--color-text-muted)", textTransform: "uppercase", marginBottom: 8, letterSpacing: "0.04em" }}>
              Configuration
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <MetricRow label="Model Type" value="RandomForestClassifier" />
              <MetricRow label="n_estimators" value="200" />
              <MetricRow label="max_depth" value="6" />
              <MetricRow label="class_weight" value="balanced" />
              <MetricRow label="CV Method" value="5-fold StratifiedKFold" />
              <MetricRow label="Scoring" value="roc_auc" />
            </div>
          </div>

          <div style={{ borderTop: "1px solid var(--color-border-light)", paddingTop: 14, marginBottom: 16 }}>
            <div style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--color-text-muted)", textTransform: "uppercase", marginBottom: 8, letterSpacing: "0.04em" }}>
              Threshold (t*_rf)
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <MetricRow label="Threshold" value="0.50" />
              <MetricRow label="Precision" value="0.3240" />
              <MetricRow label="Recall" value="0.5495" />
              <MetricRow label="F1" value="0.4076" />
            </div>
          </div>

          <div style={{ borderTop: "1px solid var(--color-border-light)", paddingTop: 14 }}>
            <div style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--color-text-muted)", textTransform: "uppercase", marginBottom: 8, letterSpacing: "0.04em" }}>
              Risk Buckets
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
              <BucketCard label="Low" range="< 0.50" color="var(--color-low)" />
              <BucketCard label="Medium" range="0.50–0.65" color="var(--color-medium)" />
              <BucketCard label="High" range="≥ 0.65" color="var(--color-high)" />
            </div>
          </div>

          <div style={{ borderTop: "1px solid var(--color-border-light)", paddingTop: 14, marginTop: 14 }}>
            <div style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--color-text-muted)", textTransform: "uppercase", marginBottom: 8, letterSpacing: "0.04em" }}>
              Top Features (Permutation Importance)
            </div>
            <FeatureBar feature="payment_method" value={0.022} maxValue={0.03} />
            <FeatureBar feature="price_inr" value={0.018} maxValue={0.03} />
            <FeatureBar feature="num_previous_returns" value={0.015} maxValue={0.03} />
            <FeatureBar feature="product_category" value={0.012} maxValue={0.03} />
            <FeatureBar feature="delivery_days" value={0.009} maxValue={0.03} />
          </div>
        </div>

        {/* Image Classifier */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">Image Classifier</div>
          </div>

          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--color-text-muted)", textTransform: "uppercase", marginBottom: 8, letterSpacing: "0.04em" }}>
              Per-Class Metrics (Test Set)
            </div>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                </tr>
              </thead>
              <tbody>
                {PER_CLASS_METRICS.map((m) => (
                  <tr key={m.class_id}>
                    <td style={{ fontWeight: 500 }}>{m.class_name}</td>
                    <td>
                      <MetricBarInline value={m.precision} />
                    </td>
                    <td>
                      <MetricBarInline value={m.recall} />
                    </td>
                    <td>
                      <MetricBarInline value={m.f1} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ borderTop: "1px solid var(--color-border-light)", paddingTop: 14 }}>
            <div style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--color-text-muted)", textTransform: "uppercase", marginBottom: 10, letterSpacing: "0.04em" }}>
              Confusion Matrix
            </div>
            <div style={{ overflowX: "auto" }}>
              <table className="data-table confusion-table">
                <thead>
                  <tr>
                    <th style={{ fontSize: "0.65rem" }}></th>
                    {CLASS_NAMES.map((n) => (
                      <th key={n} style={{ fontSize: "0.65rem" }}>{n.slice(0, 4)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {CONFUSION_MATRIX.map((row, i) => (
                    <tr key={i}>
                      <td style={{ fontWeight: 600, fontSize: "0.65rem" }}>{CLASS_NAMES[i].slice(0, 4)}</td>
                      {row.map((val, j) => (
                        <td
                          key={j}
                          className={i === j ? "diag" : ""}
                          style={{
                            background:
                              i !== j && val > 50
                                ? "var(--color-danger-bg)"
                                : i !== j && val > 20
                                ? "var(--color-warning-bg)"
                                : undefined,
                          }}
                        >
                          {val}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* RAG Retrieval */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">RAG Retrieval Evaluation</div>
          <div style={{ display: "flex", gap: 12 }}>
            <span style={{ fontSize: "0.8rem", color: "var(--color-text-muted)" }}>
              6 queries · k=3
            </span>
            <span style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--color-primary)" }}>
              Mean P@3: 0.7222 · Mean R@3: 1.0000
            </span>
          </div>
        </div>

        <table className="data-table">
          <thead>
            <tr>
              <th>Query</th>
              <th>Relevant Doc</th>
              <th>Precision@3</th>
              <th>Recall@3</th>
            </tr>
          </thead>
          <tbody>
            {RETRIEVAL_METRICS.map((m, i) => (
              <tr key={i}>
                <td style={{ fontWeight: 500, maxWidth: 300 }}>{m.query}</td>
                <td>
                  <span
                    style={{
                      padding: "2px 8px",
                      borderRadius: 4,
                      fontSize: "0.78rem",
                      background: "var(--color-primary-light)",
                      color: "var(--color-primary)",
                      fontWeight: 500,
                    }}
                  >
                    {m.relevant}
                  </span>
                </td>
                <td>
                  <MetricBarInline value={m.p3} />
                </td>
                <td>
                  <MetricBarInline value={m.r3} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* System architecture */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">System Architecture</div>
        </div>
        <div style={{ fontFamily: "monospace", fontSize: "0.78rem", lineHeight: 1.7, color: "var(--color-text-secondary)", whiteSpace: "pre-wrap" }}>
{`Customer Request
      │
      ▼
┌──────────────────────┐
│   LangGraph Agent    │
└──────────────────────┘
      │
      ▼
   Intent Node (MOCK_LLM)
      │
      ├──► Policy ──────► FAISS RAG (MiniLM-L6-v2) ──► Groundedness Check
      │                                                       │
      ├──► Return Risk ─► Random Forest Pipeline ────────────┤
      │                                                       │
      └──► Image ───────► ResNet-18 Classifier ──────────────┤
                                                               │
                                                               ▼
                                                    Response Generation
                                                               │
                                                               ▼
                                                     Structured JSON Output`}
        </div>
      </div>
    </div>
  );
}

function MetricRow({ label, value }) {
  return (
    <div style={{ padding: "4px 0" }}>
      <div style={{ fontSize: "0.72rem", color: "var(--color-text-muted)" }}>{label}</div>
      <div style={{ fontSize: "0.82rem", fontWeight: 600, marginTop: 1 }}>{value}</div>
    </div>
  );
}

function BucketCard({ label, range, color }) {
  return (
    <div style={{ padding: "8px 10px", borderRadius: "var(--radius-sm)", border: `1px solid ${color}20`, textAlign: "center" }}>
      <div style={{ fontSize: "0.78rem", fontWeight: 700, color }}>{label}</div>
      <div style={{ fontSize: "0.7rem", color: "var(--color-text-muted)" }}>{range}</div>
    </div>
  );
}

function FeatureBar({ feature, value, maxValue }) {
  const pct = (value / maxValue) * 100;
  return (
    <div className="metric-bar" style={{ marginBottom: 6 }}>
      <span style={{ fontSize: "0.75rem", minWidth: 140, color: "var(--color-text-secondary)" }}>
        {feature}
      </span>
      <div className="metric-bar-track">
        <div className="metric-bar-fill" style={{ width: `${pct}%`, background: "var(--color-primary)" }} />
      </div>
      <span className="metric-bar-value" style={{ fontSize: "0.75rem" }}>
        {value.toFixed(3)}
      </span>
    </div>
  );
}

function MetricBarInline({ value }) {
  const pct = value * 100;
  const color =
    value >= 0.9 ? "var(--color-success)" : value >= 0.8 ? "var(--color-primary)" : value >= 0.7 ? "var(--color-warning)" : "var(--color-danger)";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{ width: 50, height: 6, borderRadius: 3, background: "var(--color-border-light)", overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", borderRadius: 3, background: color }} />
      </div>
      <span style={{ fontSize: "0.75rem", fontWeight: 600, minWidth: 36 }}>
        {(value * 100).toFixed(1)}%
      </span>
    </div>
  );
}
