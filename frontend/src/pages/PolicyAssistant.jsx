import { useState, useRef, useEffect } from "react";

export default function PolicyAssistant() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async (text) => {
    const msg = (text || input).trim();
    if (!msg || loading) return;

    const userMsg = { role: "user", content: msg };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("/api/policy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: msg,
          conversation_id: conversationId,
        }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Request failed");
      }

      const data = await res.json();
      setConversationId(data.conversation_id);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.response.answer,
          source: data.response.source,
          confidence: data.response.confidence,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error: ${err.message}`,
          source: "error",
          confidence: 0,
        },
      ]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setConversationId(null);
    setInput("");
  };

  return (
    <div className="card" style={{ padding: 0, overflow: "hidden" }}>
      {/* Chat header */}
      <div
        style={{
          padding: "14px 24px",
          borderBottom: "1px solid var(--color-border)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div>
          <div style={{ fontWeight: 600, fontSize: "0.92rem" }}>
            Flipkart Policy Assistant
          </div>
          <div style={{ fontSize: "0.75rem", color: "var(--color-text-muted)", marginTop: 2 }}>
            LangGraph · FAISS RAG · Grounded Answers · Prompt Guardrails
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {conversationId && (
            <span
              style={{
                fontSize: "0.72rem",
                padding: "3px 8px",
                borderRadius: 4,
                background: "var(--color-info-bg)",
                color: "var(--color-info)",
                fontWeight: 500,
              }}
            >
              Session: {conversationId.slice(0, 8)}...
            </span>
          )}
          <button className="btn btn-secondary btn-sm" onClick={clearChat}>
            New Chat
          </button>
        </div>
      </div>

      {/* Messages area */}
      <div style={{ padding: "0 24px", maxHeight: "calc(100vh - 280px)", overflowY: "auto" }}>
        <div className="chat-messages">
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon">💬</div>
              <h3>Ask a policy question</h3>
              <p>
                Try: "How long does a COD refund take?" or "What is the return window for Apparel?"
              </p>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`chat-message ${msg.role}`}>
              <div className="chat-avatar">
                {msg.role === "user" ? "U" : "AI"}
              </div>
              <div>
                <div className="chat-bubble">{msg.content}</div>
                {msg.role === "assistant" && (
                  <ChatMeta source={msg.source} confidence={msg.confidence} />
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="chat-message assistant">
              <div className="chat-avatar">AI</div>
              <div className="chat-bubble">
                <span className="spinner" style={{ marginRight: 8 }} />
                Thinking...
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input bar */}
      <div style={{ padding: "0 24px 20px" }}>
        <div className="chat-input-bar">
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a policy question..."
            disabled={loading}
          />
          <button
            className="btn btn-primary"
            onClick={() => sendMessage()}
            disabled={loading || !input.trim()}
          >
            {loading ? <span className="spinner" /> : "Send"}
          </button>
        </div>
        <div style={{ display: "flex", gap: 8, marginTop: 8, flexWrap: "wrap" }}>
          {[
            "How long does a COD refund take?",
            "What is the return window for Apparel?",
            "When is reverse pickup available?",
            "What happens for a damaged product?",
          ].map((q) => (
            <button
              key={q}
              className="btn btn-secondary btn-sm"
              onClick={() => sendMessage(q)}
              disabled={loading}
              style={{ fontSize: "0.75rem", padding: "4px 10px" }}
            >
              {q}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function ChatMeta({ source, confidence }) {
  const sourceLabels = {
    policy_kb: "📚 Policy KB",
    return_risk_tool: "📈 Return Risk Tool",
    image_classifier_tool: "🖼️ Image Classifier",
    error: "❌ Error",
  };

  return (
    <div className="chat-meta">
      <span className="chat-meta-item chat-meta-source">
        {sourceLabels[source] || source}
      </span>
      {typeof confidence === "number" && (
        <span className="chat-meta-item chat-meta-confidence">
          Confidence: {(confidence * 100).toFixed(1)}%
        </span>
      )}
      {source === "policy_kb" && typeof confidence === "number" && (
        <span
          className={`chat-meta-item ${
            confidence >= 0.45 ? "chat-meta-grounded" : "chat-meta-not-grounded"
          }`}
        >
          {confidence >= 0.45 ? "✓ Grounded" : "✗ Not Grounded"}
        </span>
      )}
    </div>
  );
}
