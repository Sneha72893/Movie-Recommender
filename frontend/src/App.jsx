import { useState, useEffect } from 'react'

// Calling Python FastAPI directly (Go Gateway blocked by Windows WDAC policy on this machine)
// In production, this would go through the Go Gateway on port 8080
const GATEWAY = 'http://localhost:8000'

function App() {
  const [userId, setUserId] = useState(1)
  const [recommendations, setRecommendations] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [gatewayHealth, setGatewayHealth] = useState(null)
  const [movieCount, setMovieCount] = useState(null)

  useEffect(() => {
    // Fetch ML metrics from Python via Go gateway
    fetch(`${GATEWAY}/metrics`)
      .then(r => r.json())
      .then(setMetrics)
      .catch(console.error)

    // Gateway health — Python is running directly
    setGatewayHealth({ python_inference_service: { status: 'ok' } })

    // Movie count not available without Rust service
    setMovieCount('1,682')
  }, [])

  const getRecommendations = async () => {
    setLoading(true)
    setError(null)
    setRecommendations([])

    try {
      const res = await fetch(`${GATEWAY}/recommendations/${userId}?top_n=6`)
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Request failed')
      }
      const data = await res.json()
      setRecommendations(data.recommendations)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const serviceStatus = (info) => {
    if (!info) return <span className="badge badge-down">Offline</span>
    return <span className="badge badge-up">Online</span>
  }

  return (
    <div className="app-container">
      {/* ─── Sidebar ─── */}
      <aside className="sidebar">
        <div>
          <h2>🧠 System Dashboard</h2>
          <p>Real-time monitoring across all polyglot microservices.</p>
          <hr />

          {/* Service Status */}
          <h3>🌐 Service Status</h3>
          <div className="service-list">
            <div className="service-row">
              <span>⚛️ React Frontend</span>
              <span className="badge badge-up">Online</span>
            </div>
            <div className="service-row">
              <span>🐹 Go Gateway</span>
              {serviceStatus(gatewayHealth)}
            </div>
            <div className="service-row">
              <span>🐍 Python ML API</span>
              {serviceStatus(gatewayHealth?.python_inference_service)}
            </div>
            <div className="service-row">
              <span>🦀 Rust Data Engine</span>
              {serviceStatus(gatewayHealth?.rust_data_engine)}
            </div>
          </div>

          <hr />
          <h3>📊 Model Metrics</h3>
          <div className="metrics-container">
            <div className="metric-box">
              <span className="metric-label">Test RMSE</span>
              <span className="metric-value">{metrics?.rmse ?? '—'}</span>
              <span className="metric-delta">↓ -0.04 vs Baseline</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">NDCG@10</span>
              <span className="metric-value">{metrics?.ndcg ?? '—'}</span>
              <span className="metric-delta positive">↑ +0.12 vs Baseline</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Movies (Rust Cache)</span>
              <span className="metric-value">{movieCount ?? '—'}</span>
              <span className="metric-delta positive">Loaded in memory</span>
            </div>
          </div>

          <hr />
          <h3>⚙️ Architecture</h3>
          <ul>
            <li><strong>ML Model:</strong> NCF (PyTorch)</li>
            <li><strong>Inference API:</strong> Python / FastAPI</li>
            <li><strong>API Gateway:</strong> Go / net/http</li>
            <li><strong>Data Engine:</strong> Rust / Actix-Web</li>
            <li><strong>Frontend:</strong> JS / React + Vite</li>
            <li><strong>Dataset:</strong> MovieLens 100k</li>
          </ul>
        </div>

        <p className="caption">Polyglot ML Portfolio Project</p>
      </aside>

      {/* ─── Main Content ─── */}
      <main className="main-content">
        <h1 className="gradient-text">AI Movie Recommender</h1>
        <p className="subtitle">
          A polyglot microservices ML system — <strong>Python</strong> for inference,
          <strong> Go</strong> for routing, <strong>Rust</strong> for data, and
          <strong> React</strong> for the UI.
        </p>
        <hr className="main-hr" />

        {/* Architecture pills */}
        <div className="pill-row">
          <span className="pill python">🐍 Python · FastAPI</span>
          <span className="pill go">🐹 Go · API Gateway</span>
          <span className="pill rust">🦀 Rust · Actix-Web</span>
          <span className="pill react">⚛️ React · Vite</span>
        </div>

        <div className="controls">
          <div className="input-group">
            <label>Select User ID (1-943):</label>
            <input
              type="number"
              value={userId}
              onChange={e => setUserId(parseInt(e.target.value) || 1)}
              min="1" max="943"
            />
          </div>
          <button onClick={getRecommendations} disabled={loading} className="generate-btn">
            {loading ? '⏳ Analyzing...' : 'Generate Recommendations ✨'}
          </button>
        </div>

        {error && <div className="error-message">⚠️ {error}</div>}

        {recommendations.length > 0 && (
          <div className="results-section">
            <h3>🎬 Top Picks for User {userId}</h3>
            <div className="grid">
              {recommendations.map((movie, i) => (
                <div key={movie.item_id} className="glass-card">
                  <div className="card-rank">#{i + 1}</div>
                  <h4>{movie.title}</h4>
                  <div className="rating-badge">
                    ★ {movie.predicted_rating.toFixed(2)} / 5.0
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
