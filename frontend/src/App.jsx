import { useState } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5001';
const OCCASIONS = ['Casual', 'Formal', 'Party', 'Ethnic', 'Sports'];

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [occasion, setOccasion] = useState('Casual');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function handleFileChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    const formData = new FormData();
    formData.append('image', file);
    formData.append('occasion', occasion);
    try {
      const res = await fetch(`${API_URL}/analyze`, { method: 'POST', body: formData });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `Server error: ${res.status}`);
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setFile(null); setPreview(null); setResult(null); setError(null);
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-stone-50 to-stone-100 px-4 py-8 sm:py-12">
      <div className="max-w-4xl mx-auto">
        <header className="mb-10 text-center">
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-stone-900">
            FitCheck <span className="text-amber-700">AI</span>
          </h1>
          <p className="mt-3 text-stone-600">Your 24/7 personal style consultant.</p>
        </header>
        <div className="grid md:grid-cols-2 gap-8">
          <div className="bg-white rounded-2xl shadow-sm border border-stone-200 p-6">
            <h2 className="text-lg font-semibold text-stone-900 mb-4">1. Upload your outfit</h2>
            <label className="block border-2 border-dashed border-stone-300 rounded-xl cursor-pointer hover:border-amber-500 transition-colors overflow-hidden">
              {preview ? (
                <img src={preview} alt="Preview" className="w-full h-72 object-contain bg-stone-50" />
              ) : (
                <div className="h-72 flex flex-col items-center justify-center text-stone-400 p-4">
                  <svg className="w-12 h-12 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
                  </svg>
                  <span className="text-sm text-center">Click or drag a clothing photo</span>
                </div>
              )}
              <input type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
            </label>
            <div className="mt-5">
              <label className="block text-sm font-medium text-stone-700 mb-2">Occasion</label>
              <div className="flex flex-wrap gap-2">
                {OCCASIONS.map(o => (
                  <button key={o} onClick={() => setOccasion(o)}
                    className={`px-3 py-1.5 rounded-full text-sm transition ${occasion === o ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-700 hover:bg-stone-200'}`}>
                    {o}
                  </button>
                ))}
              </div>
            </div>
            <div className="mt-6 flex gap-3">
              <button onClick={analyze} disabled={!file || loading}
                className="flex-1 bg-amber-700 hover:bg-amber-800 disabled:bg-stone-300 disabled:cursor-not-allowed text-white font-medium py-3 rounded-xl transition">
                {loading ? 'Analyzing…' : 'Get my FitCheck'}
              </button>
              {file && <button onClick={reset} className="px-4 py-3 text-stone-600 hover:text-stone-900 transition">Reset</button>}
            </div>
            {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
          </div>
          <div className="bg-white rounded-2xl shadow-sm border border-stone-200 p-6">
            <h2 className="text-lg font-semibold text-stone-900 mb-4">2. Your FitCheck</h2>
            {!result && !loading && <p className="text-stone-400 text-sm">Upload an image to see your style analysis.</p>}
            {loading && (
              <div className="animate-pulse space-y-3">
                <div className="h-20 bg-stone-100 rounded-lg"></div>
                <div className="h-4 bg-stone-100 rounded w-3/4"></div>
                <div className="h-4 bg-stone-100 rounded w-1/2"></div>
                <div className="h-12 bg-stone-100 rounded"></div>
              </div>
            )}
            {result && (
              <div className="space-y-5">
                <div className="flex items-baseline gap-3">
                  <div className="text-5xl font-bold text-amber-700">{result.style_score}</div>
                  <div className="text-stone-500 text-sm">/ 100 style score</div>
                </div>
                <p className="text-stone-700 italic">"{result.rationale}"</p>
                <div>
                  <div className="text-xs uppercase tracking-wide text-stone-500 mb-1">Detected</div>
                  <div className="font-medium text-stone-900">{result.detected_category}</div>
                  <div className="text-xs text-stone-500 mt-1">Confidence: {(result.top_predictions[0].confidence * 100).toFixed(0)}%</div>
                </div>
                <div>
                  <div className="text-xs uppercase tracking-wide text-stone-500 mb-1">Dominant color</div>
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-6 rounded-full border border-stone-300" style={{ background: `rgb(${result.dominant_color.rgb.join(',')})` }} />
                    <span className="text-stone-900">{result.dominant_color.name}</span>
                  </div>
                </div>
                <div>
                  <div className="text-xs uppercase tracking-wide text-stone-500 mb-2">Pair it with</div>
                  <div className="flex flex-wrap gap-2">
                    {result.suggested_matches.map(m => (
                      <span key={m} className="px-3 py-1 bg-stone-100 text-stone-700 rounded-full text-sm">{m}</span>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
        <footer className="mt-12 text-center text-xs text-stone-400">FitCheck AI · UCP FYP · Spring 2026</footer>
      </div>
    </div>
  );
}
