'use client'

import { useState, useCallback } from 'react'
import Link from 'next/link'
import { Search, ArrowLeft, Info, TrendingUp, TrendingDown } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const KNOWN_CERTS: { cert: number; name: string }[] = [
  { cert: 628,   name: 'JPMorgan Chase' },
  { cert: 3510,  name: 'Bank of America' },
  { cert: 3511,  name: 'Wells Fargo' },
  { cert: 7213,  name: 'Citibank' },
  { cert: 12368, name: 'Regions Bank' },
  { cert: 6384,  name: 'PNC Bank' },
  { cert: 993,   name: 'Fifth Third Bank' },
  { cert: 6560,  name: 'Huntington Bank' },
]

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  const tr = payload.find((p: any) => p.name === 'Texas Ratio')
  return (
    <div className="panel-glass rounded-xl p-3 text-xs font-mono shadow-xl">
      <div className="text-ghost mb-1">{label}</div>
      {tr && <div className="text-white">Texas Ratio: <span className={tr.value > 100 ? 'text-danger' : tr.value > 50 ? 'text-warn' : 'text-signal'}>{tr.value?.toFixed(1)}</span></div>}
    </div>
  )
}

export default function BankPage() {
  const [query, setQuery] = useState('')
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const lookup = useCallback(async (cert: number) => {
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const r = await fetch(`${API}/api/bank/${cert}`)
      if (!r.ok) { setError(`CERT ${cert} not found in dataset.`); setLoading(false); return }
      setResult(await r.json())
    } catch {
      setError('API connection failed.')
    }
    setLoading(false)
  }, [])

  const handleSearch = () => {
    const n = parseInt(query.trim())
    if (isNaN(n)) { setError('Enter a valid numeric CERT.'); return }
    lookup(n)
  }

  const latest = result?.history?.[result.history.length - 1]
  const chartData = (result?.history || []).map((h: any) => ({
    date: h.date?.slice(0, 7),
    'Texas Ratio': h.texas_ratio,
    'ROA (%)': h.roa,
  }))

  const txRisk = latest?.texas_ratio
  const riskLabel = txRisk > 100 ? ['Distressed', 'text-danger'] : txRisk > 50 ? ['Elevated', 'text-warn'] : ['Healthy', 'text-signal']

  return (
    <div className="min-h-screen bg-ink noise">
      <div className="fixed inset-0 bg-grid-faint bg-grid opacity-100 pointer-events-none" />

      <nav className="fixed top-0 left-0 right-0 z-50 panel-glass border-b border-border">
        <div className="max-w-7xl mx-auto px-6 h-14 flex items-center gap-4">
          <Link href="/" className="text-ghost hover:text-bright transition-colors">
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <span className="text-ghost font-mono text-sm">/</span>
          <span className="text-bright font-mono text-sm">bank</span>
        </div>
      </nav>

      <div className="relative max-w-5xl mx-auto px-6 pt-28 pb-24">
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-blue/10">
              <Search className="w-5 h-5 text-blue" />
            </div>
            <span className="text-xs font-mono uppercase tracking-widest text-ghost">Live FDIC Lookup</span>
          </div>
          <h1 className="font-display text-4xl md:text-5xl font-bold text-white mb-4">
            Bank CERT Lookup
          </h1>
          <p className="text-ghost text-lg max-w-2xl">
            Enter any FDIC Certificate Number to see the bank&apos;s full financial history,
            Texas Ratio trajectory, and early warning distress signals.
          </p>
        </div>

        {/* Search */}
        <div className="panel-glass rounded-2xl p-6 mb-8">
          <div className="flex gap-3">
            <input
              type="text"
              placeholder="Enter CERT number (e.g. 3511)"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSearch()}
              className="flex-1 bg-ink border border-border rounded-xl px-4 py-3 text-bright font-mono text-sm placeholder-muted focus:outline-none focus:border-signal/50 transition-colors"
            />
            <button
              onClick={handleSearch}
              disabled={loading}
              className="bg-signal text-ink font-mono text-sm px-6 py-3 rounded-xl hover:bg-signal/90 transition-colors disabled:opacity-50 flex items-center gap-2"
            >
              {loading ? 'Searching...' : <><Search className="w-4 h-4" /> Search</>}
            </button>
          </div>

          {/* Quick picks */}
          <div className="mt-4 flex flex-wrap gap-2">
            <span className="text-xs font-mono text-ghost mr-2">Quick lookup:</span>
            {KNOWN_CERTS.map(({ cert, name }) => (
              <button key={cert}
                onClick={() => { setQuery(cert.toString()); lookup(cert) }}
                className="text-xs font-mono text-ghost border border-border px-3 py-1.5 rounded-full hover:border-signal/30 hover:text-signal transition-colors">
                {name} ({cert})
              </button>
            ))}
          </div>

          {error && (
            <div className="mt-4 text-danger text-xs font-mono p-3 border border-danger/20 rounded-lg bg-danger/5">
              {error}
            </div>
          )}
        </div>

        {/* Result */}
        {result && (
          <>
            <div className="panel-glass rounded-2xl p-8 mb-6">
              <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4 mb-8">
                <div>
                  <div className="font-display text-3xl text-white font-bold mb-1">
                    {result.name || `CERT ${result.cert}`}
                  </div>
                  <div className="text-ghost font-mono text-sm">
                    {[result.city, result.state].filter(Boolean).join(', ')} · CERT {result.cert}
                  </div>
                </div>
                {latest && (
                  <div className={`text-sm font-mono border px-4 py-2 rounded-full self-start ${
                    riskLabel[1] === 'text-danger' ? 'bg-danger/10 text-danger border-danger/20' :
                    riskLabel[1] === 'text-warn'   ? 'bg-warn/10 text-warn border-warn/20' :
                                                     'bg-signal/10 text-signal border-signal/20'
                  }`}>
                    {riskLabel[0]}
                  </div>
                )}
              </div>

              {latest && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                  {[
                    { label: 'Texas Ratio', value: latest.texas_ratio?.toFixed(1) ?? '—', color: latest.texas_ratio > 100 ? 'text-danger' : latest.texas_ratio > 50 ? 'text-warn' : 'text-signal' },
                    { label: 'Total Assets ($M)', value: latest.assets_m ? `$${(latest.assets_m).toLocaleString(undefined, {maximumFractionDigits: 0})}` : '—', color: 'text-bright' },
                    { label: 'ROA (%)', value: latest.roa?.toFixed(2) ?? '—', color: (latest.roa || 0) > 0 ? 'text-signal' : 'text-danger' },
                    { label: 'As of', value: latest.date?.slice(0, 7) ?? '—', color: 'text-ghost' },
                  ].map(({ label, value, color }) => (
                    <div key={label} className="bg-white/[0.03] rounded-xl p-4">
                      <div className={`font-display text-xl font-bold ${color}`}>{value}</div>
                      <div className="text-ghost text-xs mt-1 font-mono">{label}</div>
                    </div>
                  ))}
                </div>
              )}

              {chartData.length > 0 && (
                <>
                  <div className="text-bright text-sm font-mono mb-4">Texas Ratio — Full History</div>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={chartData}>
                      <XAxis dataKey="date" tick={{ fill: '#6b7280', fontSize: 9, fontFamily: 'monospace' }} interval="preserveStartEnd" />
                      <YAxis tick={{ fill: '#6b7280', fontSize: 9, fontFamily: 'monospace' }} />
                      <Tooltip content={<CustomTooltip />} />
                      <ReferenceLine y={100} stroke="#ef4444" strokeDasharray="4 4" strokeOpacity={0.5} label={{ value: 'Distress threshold (100)', fill: '#ef4444', fontSize: 9 }} />
                      <Line type="monotone" dataKey="Texas Ratio" stroke="#00d4aa" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </>
              )}
            </div>

            <div className="flex items-start gap-3 text-ghost text-xs font-mono p-4 border border-border/50 rounded-xl">
              <Info className="w-4 h-4 shrink-0 mt-0.5" />
              <span>
                Texas Ratio = non-current loans / (tangible equity + loan loss reserves) × 100.
                Values above 100 have historically preceded bank failure. Data from FDIC SDI API,
                Q1 2005 to Q4 2025.
              </span>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
