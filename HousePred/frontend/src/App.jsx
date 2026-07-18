import { useState } from 'react'

const API_URL = 'http://localhost:8000/predict'

const QUALITY_LEVELS = [
  { value: 10, label: 'Very Excellent' },
  { value: 9, label: 'Excellent' },
  { value: 8, label: 'Very Good' },
  { value: 7, label: 'Good' },
  { value: 6, label: 'Above Average' },
  { value: 5, label: 'Average' },
  { value: 4, label: 'Below Average' },
  { value: 3, label: 'Fair' },
  { value: 2, label: 'Poor' },
  { value: 1, label: 'Very Poor' },
]

const BATH_OPTIONS = [0, 1, 2, 3, 4, 5]
const GARAGE_OPTIONS = [0, 1, 2, 3, 4]
const ROOM_OPTIONS = Array.from({ length: 20 }, (_, i) => i + 1)

const FIELDS = [
  { name: 'OverallQual', label: 'Overall Quality', hint: 'Material & finish quality', type: 'select', options: QUALITY_LEVELS },
  { name: 'GrLivArea', label: 'Living Area', hint: 'Above-ground, in sq ft', type: 'number', min: 1 },
  { name: 'GarageCars', label: 'Garage Capacity', hint: 'Number of cars', type: 'select', options: GARAGE_OPTIONS.map((v) => ({ value: v, label: `${v} car${v === 1 ? '' : 's'}` })) },
  { name: 'TotalBsmtSF', label: 'Basement Area', hint: 'Total, in sq ft', type: 'number', min: 0 },
  { name: 'FullBath', label: 'Full Bathrooms', hint: 'Count', type: 'select', options: BATH_OPTIONS.map((v) => ({ value: v, label: String(v) })) },
  { name: 'TotRmsAbvGrd', label: 'Total Rooms', hint: 'Above ground, excl. bathrooms', type: 'select', options: ROOM_OPTIONS.map((v) => ({ value: v, label: String(v) })) },
  { name: 'YearBuilt', label: 'Year Built', hint: '', type: 'number', min: 1870, max: 2026 },
  { name: 'YearRemodAdd', label: 'Year Remodeled', hint: 'Same as build year if never remodeled', type: 'number', min: 1870, max: 2026 },
]

const initialForm = FIELDS.reduce((acc, f) => ({ ...acc, [f.name]: '' }), {})

function money(n) {
  return `$${Math.round(n).toLocaleString()}`
}

export default function App() {
  const [form, setForm] = useState(initialForm)
  const [result, setResult] = useState(null) // { predicted_price, baseline_price, explanation }
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleChange = (name, value) => {
    setForm((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    setResult(null)

    const missing = FIELDS.filter((f) => form[f.name] === '')
    if (missing.length > 0) {
      setError(`Please fill in: ${missing.map((f) => f.label).join(', ')}`)
      return
    }

    const payload = {}
    FIELDS.forEach((f) => {
      payload[f.name] = Number(form[f.name])
    })

    setLoading(true)
    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!res.ok) {
        const errBody = await res.json().catch(() => null)
        throw new Error(errBody?.detail || `Request failed (${res.status})`)
      }

      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Something went wrong. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setForm(initialForm)
    setResult(null)
    setError(null)
  }

  const maxAbsImpact = result
    ? Math.max(...result.explanation.map((f) => Math.abs(f.impact)), 1)
    : 1

  return (
    <div className="app">
      <div className="grid-backdrop" />

      <main className="card">
        <header className="card-header">
          <div className="header-mark">⌂</div>
          <div>
            <h1>House Price Estimator</h1>
            <p>Fill in the key details — everything else is filled in from typical values.</p>
          </div>
        </header>

        <form onSubmit={handleSubmit} className="form-grid">
          {FIELDS.map((f) => (
            <label key={f.name} className="field">
              <span className="field-label">
                {f.label}
                {f.hint && <span className="field-hint"> · {f.hint}</span>}
              </span>

              {f.type === 'select' ? (
                <select value={form[f.name]} onChange={(e) => handleChange(f.name, e.target.value)}>
                  <option value="" disabled>Select…</option>
                  {f.options.map((opt) => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  min={f.min}
                  max={f.max}
                  value={form[f.name]}
                  onChange={(e) => handleChange(f.name, e.target.value)}
                  placeholder="—"
                />
              )}
            </label>
          ))}

          <div className="actions">
            <button type="submit" disabled={loading}>
              {loading ? 'Estimating…' : 'Estimate Price'}
            </button>
            <button type="button" className="secondary" onClick={handleReset}>
              Reset
            </button>
          </div>
        </form>

        {error && <div className="error-box">{error}</div>}

        {result && (
          <>
            <div className="result-box">
              <span className="result-label">Estimated Price</span>
              <span className="result-value">{money(result.predicted_price)}</span>
              <span className="result-note">
                Typical house in this dataset: {money(result.baseline_price)} · ElasticNet regression model
              </span>
            </div>

            <div className="explain-box">
              <h2>Why this price?</h2>
              <p className="explain-sub">
                Effect of each field, compared to a typical house — everything else held constant.
              </p>

              <ul className="explain-list">
                {result.explanation.map((f) => {
                  const pct = (Math.abs(f.impact) / maxAbsImpact) * 100
                  const positive = f.impact >= 0
                  return (
                    <li key={f.feature} className="explain-row">
                      <div className="explain-row-top">
                        <span className="explain-label">{f.label}</span>
                        <span className={`explain-amount ${positive ? 'up' : 'down'}`}>
                          {positive ? '+' : '−'}{money(Math.abs(f.impact))}
                        </span>
                      </div>
                      <div className="explain-bar-track">
                        <div
                          className={`explain-bar-fill ${positive ? 'up' : 'down'}`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </li>
                  )
                })}
              </ul>
            </div>
          </>
        )}
      </main>
    </div>
  )
}