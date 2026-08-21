import { useState, useRef } from "react";

const ALLOWED_TYPES = ["image/png", "image/jpeg", "image/jpg"];
const ALLOWED_EXTENSIONS = [".png", ".jpg", ".jpeg"];

export default function ImageClassifier() {
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [selectedSample, setSelectedSample] = useState(null);
  const fileInputRef = useRef(null);

  const SAMPLE_IMAGES = [
    { name: "test_0000_true_9_Ankle_boot.png", label: "Ankle Boot" },
    { name: "test_0001_true_2_Pullover.png", label: "Pullover" },
    { name: "test_0002_true_1_Trouser.png", label: "Trouser" },
    { name: "test_0003_true_1_Trouser.png", label: "Trouser" },
    { name: "test_0004_true_6_Shirt.png", label: "Shirt" },
  ];

  const handleFile = (f) => {
    setError(null);
    setResult(null);

    if (!f) return;

    if (!ALLOWED_TYPES.includes(f.type)) {
      setError("Unsupported file type. Please use PNG, JPG, or JPEG.");
      return;
    }

    const ext = "." + f.name.split(".").pop().toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setError("Unsupported file extension. Please use .png, .jpg, or .jpeg");
      return;
    }

    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    handleFile(f);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleFileInput = (e) => {
    handleFile(e.target.files[0]);
  };

  const handleSampleImage = async (sampleName) => {
    setError(null);
    setResult(null);
    setFile(null);
    setPreview(null);
    setLoading(true);
    setSelectedSample(sampleName);

    try {
      const res = await fetch(`/api/sample-images/${sampleName}`);
      if (!res.ok) throw new Error("Failed to load sample image");

      const blob = await res.blob();
      const ext = "." + sampleName.split(".").pop().toLowerCase();
      const mime = ext === ".png" ? "image/png" : "image/jpeg";
      const imageFile = new File([blob], sampleName, { type: mime });

      setFile(imageFile);
      setPreview(URL.createObjectURL(blob));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const removeImage = () => {
    setPreview(null);
    setFile(null);
    setResult(null);
    setError(null);
    setSelectedSample(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const classify = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("/api/classify-image", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Classification failed");
      }

      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, alignItems: "start" }}>
      {/* Upload area */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Upload Product Image</div>
            <div className="card-subtitle">
              Drag & drop or select a Fashion-MNIST image
            </div>
          </div>
        </div>

        {!preview ? (
          <div
            className={`upload-zone ${dragOver ? "dragover" : ""}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="upload-icon">📷</div>
            <div className="upload-text">
              Drag & drop an image here, or click to browse
            </div>
            <div className="upload-hint">
              Supports PNG, JPG, JPEG — Max 10 MB
            </div>
          </div>
        ) : (
          <div style={{ textAlign: "center", padding: "16px 0" }}>
            <div className="image-preview">
              <img src={preview} alt="Preview" />
              <button className="remove-btn" onClick={removeImage}>
                ×
              </button>
            </div>
            <div style={{ marginTop: 10, fontSize: "0.8rem", color: "var(--color-text-muted)" }}>
              {file?.name} ({(file?.size / 1024).toFixed(1)} KB)
            </div>
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept=".png,.jpg,.jpeg"
          onChange={handleFileInput}
          style={{ display: "none" }}
        />

        <div style={{ marginTop: 16, display: "flex", gap: 10 }}>
          <button
            className="btn btn-primary"
            disabled={!file || loading}
            onClick={classify}
            style={{ flex: 1 }}
          >
            {loading ? (
              <>
                <span className="spinner" /> Classifying...
              </>
            ) : (
              "Classify Image"
            )}
          </button>
          {preview && (
            <button
              className="btn btn-secondary"
              onClick={removeImage}
              disabled={loading}
            >
              Remove
            </button>
          )}
        </div>

        {/* Sample images */}
        <div style={{ marginTop: 16, borderTop: "1px solid var(--color-border-light)", paddingTop: 14 }}>
          <div style={{ fontSize: "0.78rem", color: "var(--color-text-muted)", marginBottom: 10 }}>
            Or try a sample image:
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {SAMPLE_IMAGES.map((sample) => (
              <button
                key={sample.name}
                onClick={() => handleSampleImage(sample.name)}
                disabled={loading}
                style={{
                  fontSize: "0.76rem",
                  padding: "6px 12px",
                  borderRadius: 6,
                  border: selectedSample === sample.name ? "2px solid var(--color-primary)" : "1px solid var(--color-border)",
                  background: selectedSample === sample.name ? "var(--color-primary-light)" : "var(--color-surface)",
                  color: selectedSample === sample.name ? "var(--color-primary)" : "var(--color-text-secondary)",
                  cursor: loading ? "not-allowed" : "pointer",
                  fontWeight: 500,
                  transition: "all 0.15s ease",
                  opacity: loading && selectedSample !== sample.name ? 0.6 : 1,
                }}
              >
                {sample.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Results */}
      <div>
        {error && (
          <div
            className="card"
            style={{ borderLeft: "4px solid var(--color-danger)", marginBottom: 16 }}
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
              Running ResNet-18 classifier...
            </div>
          </div>
        )}

        {!result && !loading && !error && (
          <div className="card">
            <div className="empty-state">
              <div className="empty-icon">🖼️</div>
              <h3>No Classification Yet</h3>
              <p>Upload an image and click "Classify Image"</p>
            </div>
          </div>
        )}

        {result && (
          <div className="card">
            <div className="card-header">
              <div className="card-title">Classification Result</div>
            </div>

            {/* Category */}
            <div style={{ textAlign: "center", padding: "20px 0" }}>
              <div style={{ fontSize: "2.2rem", fontWeight: 700, color: "var(--color-primary)", marginBottom: 4 }}>
                {result.category}
              </div>
              <div style={{ fontSize: "0.85rem", color: "var(--color-text-muted)" }}>
                Predicted Category
              </div>
            </div>

            {/* Confidence */}
            <div style={{ marginBottom: 20 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: "0.8rem", color: "var(--color-text-muted)" }}>
                  Confidence
                </span>
                <span style={{ fontSize: "0.88rem", fontWeight: 700 }}>
                  {(result.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="probability-gauge">
                <div
                  className="probability-gauge-fill"
                  style={{
                    width: `${result.confidence * 100}%`,
                    backgroundColor:
                      result.confidence >= 0.8
                        ? "var(--color-success)"
                        : result.confidence >= 0.5
                        ? "var(--color-warning)"
                        : "var(--color-danger)",
                  }}
                />
              </div>
            </div>

            {/* Model info */}
            <div style={{ borderTop: "1px solid var(--color-border-light)", paddingTop: 14 }}>
              <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--color-text-muted)", textTransform: "uppercase", marginBottom: 8, letterSpacing: "0.04em" }}>
                Model Details
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                <DetailItem label="Architecture" value="ResNet-18" />
                <DetailItem label="Dataset" value="Fashion-MNIST" />
                <DetailItem label="Test Accuracy" value="88.49%" />
                <DetailItem label="Classes" value="10" />
              </div>
            </div>

            {/* All classes */}
            <div style={{ borderTop: "1px solid var(--color-border-light)", paddingTop: 14, marginTop: 14 }}>
              <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--color-text-muted)", textTransform: "uppercase", marginBottom: 8, letterSpacing: "0.04em" }}>
                Supported Classes
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {[
                  "T-shirt_top", "Trouser", "Pullover", "Dress", "Coat",
                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot",
                ].map((c) => (
                  <span
                    key={c}
                    style={{
                      padding: "4px 10px",
                      borderRadius: 4,
                      fontSize: "0.75rem",
                      fontWeight: 500,
                      background: c === result.category ? "var(--color-primary)" : "var(--color-bg)",
                      color: c === result.category ? "#fff" : "var(--color-text-muted)",
                    }}
                  >
                    {c}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function DetailItem({ label, value }) {
  return (
    <div style={{ padding: "4px 0" }}>
      <div style={{ fontSize: "0.72rem", color: "var(--color-text-muted)" }}>{label}</div>
      <div style={{ fontSize: "0.82rem", fontWeight: 600, marginTop: 1 }}>{value}</div>
    </div>
  );
}
