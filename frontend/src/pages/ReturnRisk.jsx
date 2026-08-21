import { useState } from "react";

const DEFAULT_VALUES = {
  product_category: "Apparel",
  price_inr: 1200,
  discount_pct: 25,
  payment_method: "COD",
  customer_tenure_days: 300,
  num_previous_orders: 8,
  num_previous_returns: 2,
  delivery_distance_km: 150,
  delivery_days: 5,
  is_weekend_order: 0,
  rating_given: 3,
};

const CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"];
const PAYMENT_METHODS = ["COD", "Prepaid_Card", "Prepaid_UPI", "Wallet"];

export default function ReturnRisk() {
  const [form, setForm] = useState(DEFAULT_VALUES);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: ["price_inr", "discount_pct", "delivery_distance_km", "rating_given"].includes(name)
        ? parseFloat(value) || 0
        : ["customer_tenure_days", "num_previous_orders", "num_previous_returns", "delivery_days"].includes(name)
        ? parseInt(value, 10) || 0
        : value,
    }));
  };

  const handleWeekendToggle = () => {
    setForm((prev) => ({ ...prev, is_weekend_order: prev.is_weekend_order ? 0 : 1 }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("/api/return-risk", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Request failed");
      }

      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const probPct = result ? Math.round(result.return_probability * 100) : 0;
  const gaugeColor = result
    ? result.risk_bucket === "High"
      ? "var(--color-high)"
      : result.risk_bucket === "Medium"
      ? "var(--color-medium)"
      : "var(--color-low)"
    : "var(--color-border)";

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, alignItems: "start" }}>
      {/* Form */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Order Features</div>
            <div className="card-subtitle">
              Enter order details to predict return risk
            </div>
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="form-row" style={{ marginBottom: 14 }}>
            <div className="form-group">
              <label className="form-label">Product Category</label>
              <select
                name="product_category"
                value={form.product_category}
                onChange={handleChange}
                className="form-select"
              >
                {CATEGORIES.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Payment Method</label>
              <select
                name="payment_method"
                value={form.payment_method}
                onChange={handleChange}
                className="form-select"
              >
                {PAYMENT_METHODS.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="form-row" style={{ marginBottom: 14 }}>
            <div className="form-group">
              <label className="form-label">Price (₹)</label>
              <input
                type="number"
                name="price_inr"
                value={form.price_inr}
                onChange={handleChange}
                className="form-input"
                min="0"
                step="1"
              />
            </div>
            <div className="form-group">
              <label className="form-label">Discount %</label>
              <input
                type="number"
                name="discount_pct"
                value={form.discount_pct}
                onChange={handleChange}
                className="form-input"
                min="0"
                max="100"
                step="0.1"
              />
            </div>
          </div>

          <div className="form-row" style={{ marginBottom: 14 }}>
            <div className="form-group">
              <label className="form-label">Customer Tenure (days)</label>
              <input
                type="number"
                name="customer_tenure_days"
                value={form.customer_tenure_days}
                onChange={handleChange}
                className="form-input"
                min="0"
              />
            </div>
            <div className="form-group">
              <label className="form-label">Previous Orders</label>
              <input
                type="number"
                name="num_previous_orders"
                value={form.num_previous_orders}
                onChange={handleChange}
                className="form-input"
                min="0"
              />
            </div>
          </div>

          <div className="form-row" style={{ marginBottom: 14 }}>
            <div className="form-group">
              <label className="form-label">Previous Returns</label>
              <input
                type="number"
                name="num_previous_returns"
                value={form.num_previous_returns}
                onChange={handleChange}
                className="form-input"
                min="0"
              />
            </div>
            <div className="form-group">
              <label className="form-label">Delivery Distance (km)</label>
              <input
                type="number"
                name="delivery_distance_km"
                value={form.delivery_distance_km}
                onChange={handleChange}
                className="form-input"
                min="0"
                step="0.1"
              />
            </div>
          </div>

          <div className="form-row" style={{ marginBottom: 14 }}>
            <div className="form-group">
              <label className="form-label">Delivery Days</label>
              <input
                type="number"
                name="delivery_days"
                value={form.delivery_days}
                onChange={handleChange}
                className="form-input"
                min="1"
              />
            </div>
            <div className="form-group">
              <label className="form-label">Rating Given</label>
              <input
                type="number"
                name="rating_given"
                value={form.rating_given ?? ""}
                onChange={handleChange}
                className="form-input"
                min="1"
                max="5"
                step="1"
                placeholder="Optional"
              />
            </div>
          </div>

          <div style={{ marginBottom: 18 }}>
            <button
              type="button"
              onClick={handleWeekendToggle}
              className={`btn btn-sm ${form.is_weekend_order ? "btn-primary" : "btn-secondary"}`}
              style={{ marginRight: 8 }}
            >
              {form.is_weekend_order ? "✓ Weekend Order" : "Weekend Order"}
            </button>
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading}
            style={{ width: "100%" }}
          >
            {loading ? (
              <>
                <span className="spinner" /> Predicting...
              </>
            ) : (
              "Predict Return Risk"
            )}
          </button>
        </form>
      </div>

      {/* Results */}
      <div>
        {error && (
          <div
            className="card"
            style={{
              borderLeft: "4px solid var(--color-danger)",
              marginBottom: 16,
            }}
          >
            <div style={{ color: "var(--color-danger)", fontWeight: 600, fontSize: "0.88rem" }}>
              Error
            </div>
            <div style={{ fontSize: "0.85rem", color: "var(--color-text-secondary)", marginTop: 4 }}>
              {error}
            </div>
          </div>
        )}

        {loading && (
          <div className="card">
            <div className="loading-overlay">
              <span className="spinner spinner-lg" />
              Running Random Forest prediction...
            </div>
          </div>
        )}

        {!result && !loading && !error && (
          <div className="card">
            <div className="empty-state">
              <div className="empty-icon">📈</div>
              <h3>No Prediction Yet</h3>
              <p>Fill in the order details and click "Predict Return Risk"</p>
            </div>
          </div>
        )}

        {result && (
          <div className="card" style={{ borderLeft: `4px solid ${gaugeColor}` }}>
            <div className="card-header">
              <div className="card-title">Prediction Result</div>
              <span className={`risk-badge risk-${result.risk_bucket.toLowerCase()}`}>
                {result.risk_bucket} Risk
              </span>
            </div>

            {/* Probability gauge */}
            <div style={{ marginBottom: 20 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: "0.8rem", color: "var(--color-text-muted)" }}>
                  Return Probability
                </span>
                <span style={{ fontSize: "0.88rem", fontWeight: 700 }}>
                  {(result.return_probability * 100).toFixed(1)}%
                </span>
              </div>
              <div className="probability-gauge">
                <div
                  className="probability-gauge-fill"
                  style={{
                    width: `${probPct}%`,
                    backgroundColor: gaugeColor,
                  }}
                />
              </div>
            </div>

            {/* Threshold markers */}
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "var(--color-text-muted)", marginBottom: 24, padding: "0 2px" }}>
              <span>0%</span>
              <span style={{ color: "var(--color-medium)", fontWeight: 600 }}>
                Threshold: {(result.threshold * 100).toFixed(0)}%
              </span>
              <span style={{ color: "var(--color-high)", fontWeight: 600 }}>
                High: {(result.high_cutoff * 100).toFixed(0)}%
              </span>
              <span>100%</span>
            </div>

            {/* Risk buckets */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 20 }}>
              <RiskBucket
                label="Low"
                range={`< ${(result.threshold * 100).toFixed(0)}%`}
                active={result.risk_bucket === "Low"}
              />
              <RiskBucket
                label="Medium"
                range={`${(result.threshold * 100).toFixed(0)}%–${(result.high_cutoff * 100).toFixed(0)}%`}
                active={result.risk_bucket === "Medium"}
              />
              <RiskBucket
                label="High"
                range={`≥ ${(result.high_cutoff * 100).toFixed(0)}%`}
                active={result.risk_bucket === "High"}
              />
            </div>

            {/* Details */}
            <div style={{ borderTop: "1px solid var(--color-border-light)", paddingTop: 14 }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                <DetailItem label="Model Type" value="RandomForestClassifier" />
                <DetailItem label="Best Parameters" value="n_estimators=200, max_depth=6" />
                <DetailItem label="Test ROC-AUC" value={result.threshold ? "0.6203" : "—"} />
                <DetailItem label="CV ROC-AUC" value={result.threshold ? "0.6192" : "—"} />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function RiskBucket({ label, range, active }) {
  const color =
    label === "Low" ? "var(--color-low)" : label === "Medium" ? "var(--color-medium)" : "var(--color-high)";
  const bg =
    label === "Low" ? "var(--color-low-bg)" : label === "Medium" ? "var(--color-medium-bg)" : "var(--color-high-bg)";

  return (
    <div
      style={{
        padding: "10px 12px",
        borderRadius: "var(--radius-sm)",
        border: active ? `2px solid ${color}` : "1px solid var(--color-border-light)",
        background: active ? bg : "transparent",
        textAlign: "center",
      }}
    >
      <div style={{ fontSize: "0.82rem", fontWeight: 700, color }}>{label}</div>
      <div style={{ fontSize: "0.72rem", color: "var(--color-text-muted)" }}>{range}</div>
    </div>
  );
}

function DetailItem({ label, value }) {
  return (
    <div style={{ padding: "6px 0" }}>
      <div style={{ fontSize: "0.72rem", color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
        {label}
      </div>
      <div style={{ fontSize: "0.82rem", fontWeight: 600, marginTop: 2 }}>{value}</div>
    </div>
  );
}
