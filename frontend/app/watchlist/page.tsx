'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { AlertTriangle, ArrowLeft, TrendingUp, Info } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { Logo, BackToTop, ScrollProgress } from '@/components/shared'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

function getRiskColor(prob: number) {
  if (prob >= 0.15) return '#ef4444'
  if (prob >= 0.05) return '#f59e0b'
  return '#00d4aa'
}

function RiskBadge({ prob }: { prob: number }) {
  const label = prob >= 0.15 ? 'High' : prob >= 0.05 ? 'Elevated' : 'Watch'
  const styles = prob >= 0.15
    ? 'bg-danger/10 text-danger border-danger/20'
    : prob >= 0.05
    ? 'bg-warn/10 text-warn border-warn/20'
    : 'bg-signal/10 text-signal border-signal/20'
  return <span className={`text-xs font-mono border px-2.5 py-1 rounded-full ${styles}`}>{label}</span>
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="panel-glass rounded-xl p-3 text-xs font-mono shadow-xl">
      <div className="text-ghost mb-1">CERT {label}</div>
      <div className="text-white">P(distress): <span className="text-signal">{(payload[0].value * 100).toFixed(1)}%</span></div>
    </div>
  )
}

export default function WatchlistPage() {
  const [banks, setBanks] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<'all' | 'high' | 'elevated'>('all')

  useEffect(() => {
    fetch(`${API}/api/watchlist?limit=100`)
      .then(r => r.json())
      .then(d => { setBanks(d.data || []); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const filtered = banks.filter(b => {
    if (filter === 'high') return (b.distress_prob || 0) >= 0.15
    if (filter === 'elevated') return (b.distress_prob || 0) >= 0.05
    return true
  })

  const chartData = banks.slice(0, 20).map(b => ({
    name: b.CERT?.toString() || '—',
    prob: b.distress_prob || 0,
  }))

  return (
    <div className="min-h-screen bg-ink noise">
      <ScrollProgress />
      <div className="fixed inset-0 bg-grid-faint bg-grid opacity-100 pointer-events-none" />
      <BackToTop />

      <nav className="fixed top-0 left-0 right-0 z-50 panel-glass border-b border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center gap-3">
          <Link href="/" className="text-ghost hover:text-bright transition-colors p-1">
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <span className="text-ghost font-mono text-sm">/</span>
          <span className="text-bright font-mono text-sm">watchlist</span>
          <div className="ml-auto">
            <Link href="/" className="flex items-center gap-2">
              <Logo size={20} />
            </Link>
          </div>
        </div>
      </nav>

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 pt-24 sm:pt-28 pb-24">
        <div className="mb-10 sm:mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-warn/10">
              <AlertTriangle className="w-5 h-5 text-warn" />
            </div>
            <span className="text-xs font-mono uppercase tracking-widest text-ghost">Early Warning System</span>
          </div>
          <h1 className="font-display text-3xl sm:text-4xl md:text-5xl font-bold text-white mb-4">
            Bank Distress Watchlist
          </h1>
          <p className="text-ghost text-base sm:text-lg max-w-2xl">
            XGBoost classifier trained on 435,623 bank-quarters (Q1 2005 to Q4 2019).
            Out-of-sample AUC = 0.804. Precision@50 = 0.100 — 100-fold lift over base rate.
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4 mb-10 sm:mb-12">
          {[
            { label: 'AUC (test)', value: '0.804', color: 'text-signal' },
            { label: 'Precision@50', value: '10.0%', color: 'text-blue' },
            { label: 'Lift vs random', value: '100×', color: 'text-warn' },
            { label: 'Top feature', value: 'Texas Ratio', color: 'text-bright' },
          ].map(({ label, value, color }) => (
            <div key={label} className="panel-glass rounded-xl p-4 sm:p-5 hover:border-white/10 transition-all">
              <div className={`font-display text-xl sm:text-2xl font-bold ${color}`}>{value}</div>
              <div className="text-ghost text-xs mt-1 font-mono">{label}</div>
            </div>
          ))}
        </div>

        {/* Chart */}
        {chartData.length > 0 && (
          <div className="panel-glass rounded-2xl p-4 sm:p-6 mb-8 sm:mb-10">
            <div className="flex items-center gap-2 mb-5 sm:mb-6">
              <TrendingUp className="w-4 h-4 text-signal" />
              <span className="text-bright text-sm font-mono">Top 20 by distress probability</span>
            </div>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={chartData} barSize={16}>
                <XAxis dataKey="name" tick={{ fill: '#6b7280', fontSize: 9, fontFamily: 'monospace' }} />
                <YAxis tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} tick={{ fill: '#6b7280', fontSize: 9, fontFamily: 'monospace' }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="prob" radius={[4, 4, 0, 0]}>
                  {chartData.map((entry, i) => (
                    <Cell key={i} fill={getRiskColor(entry.prob)} fillOpacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Filter tabs */}
        <div className="flex gap-2 mb-6">
          {(['all', 'elevated', 'high'] as const).map(f => (
            <button key={f} onClick={() => setFilter(f)}
              className={`text-xs font-mono px-4 py-2 rounded-full border transition-all ${
                filter === f
                  ? 'bg-signal text-ink border-signal'
                  : 'border-border text-ghost hover:border-signal/30 hover:text-signal'
              }`}>
              {f === 'all' ? 'All banks' : f === 'elevated' ? 'z > 5%' : 'High risk'}
            </button>
          ))}
          <span className="ml-auto text-xs font-mono text-ghost self-center">
            {filtered.length} institutions
          </span>
        </div>

        {/* Table */}
        {loading ? (
          <div className="panel-glass rounded-2xl p-12 text-center text-ghost font-mono animate-pulse">
            Loading watchlist data...
          </div>
        ) : filtered.length === 0 ? (
          <div className="panel-glass rounded-2xl p-12 text-center text-ghost font-mono">
            No data available.
          </div>
        ) : (
          <div className="panel-glass rounded-2xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    {['CERT', 'Location', 'P(distress)', 'Texas Ratio', 'ROA', 'Risk'].map((h, i) => (
                      <th key={h} className={`p-3 sm:p-4 text-ghost font-mono text-xs uppercase tracking-widest ${i > 1 ? 'text-right' : 'text-left'}`}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((b, i) => (
                    <tr key={i} className="border-b border-border/50 hover:bg-white/[0.02] transition-colors">
                      <td className="p-3 sm:p-4 font-mono text-bright">
                        <Link href={`/bank?cert=${b.CERT}`} className="hover:text-signal transition-colors">
                          {b.CERT}
                        </Link>
                      </td>
                      <td className="p-3 sm:p-4 text-ghost text-xs">{[b.CITY, b.STNAME].filter(Boolean).join(', ') || '—'}</td>
                      <td className="p-3 sm:p-4 text-right font-mono">
                        <span style={{ color: getRiskColor(b.distress_prob || 0) }}>
                          {((b.distress_prob || 0) * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="p-3 sm:p-4 text-right font-mono text-ghost text-xs">
                        {b.texas_ratio != null ? b.texas_ratio.toFixed(1) : '—'}
                      </td>
                      <td className="p-3 sm:p-4 text-right font-mono text-ghost text-xs">
                        {b.roa != null ? b.roa.toFixed(2) : '—'}
                      </td>
                      <td className="p-3 sm:p-4 text-right">
                        <RiskBadge prob={b.distress_prob || 0} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        <div className="mt-8 flex items-start gap-3 text-ghost text-xs font-mono p-4 border border-border/50 rounded-xl">
          <Info className="w-4 h-4 shrink-0 mt-0.5" />
          <span>Distress probability is the Platt-calibrated XGBoost output predicting Texas Ratio exceeding 100 within four quarters. For informational and research purposes only.</span>
        </div>
      </div>
    </div>
  )
}