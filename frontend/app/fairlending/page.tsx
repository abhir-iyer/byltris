'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { Shield, ArrowLeft, Info } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts'
import { Logo, BackToTop, ScrollProgress } from '@/components/shared'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  const or = payload[0].value as number
  return (
    <div className="panel-glass rounded-xl p-3 text-xs font-mono shadow-xl">
      <div className="text-bright font-medium mb-1">{label}</div>
      <div className="text-ghost">Black-White OR: <span className="text-white">{or.toFixed(3)}</span></div>
      <div className="text-ghost">Gap from parity: <span className="text-danger">{((1 - or) * 100).toFixed(1)}%</span></div>
    </div>
  )
}

export default function FairLendingPage() {
  const [data, setData] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`${API}/api/fairlending`)
      .then(r => r.json())
      .then(d => { setData(d.data || []); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const chartData = [...data]
    .sort((a, b) => (a.black_OR || 0) - (b.black_OR || 0))
    .map(d => ({
      name: (d.institution || d.name || '').replace(' Bank', '').replace(' Financial', ''),
      or: d.black_OR || 0,
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
          <span className="text-bright font-mono text-sm">fair-lending</span>
          <div className="ml-auto">
            <Link href="/"><Logo size={20} /></Link>
          </div>
        </div>
      </nav>

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 pt-24 sm:pt-28 pb-24">
        <div className="mb-10 sm:mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-blue/10">
              <Shield className="w-5 h-5 text-blue" />
            </div>
            <span className="text-xs font-mono uppercase tracking-widest text-ghost">HMDA Analysis · 2021–2023</span>
          </div>
          <h1 className="font-display text-3xl sm:text-4xl md:text-5xl font-bold text-white mb-4">
            Fair Lending Analysis
          </h1>
          <p className="text-ghost text-base sm:text-lg max-w-2xl">
            Two-stage logistic regression on 4.49 million HMDA mortgage applications across ten institutions.
            Black-White adjusted approval odds ratios after controlling for income, DTI, loan amount, and purpose.
          </p>
        </div>

        {/* Key numbers — neutral, industry-wide */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4 mb-10 sm:mb-12">
          {[
            { label: 'Industry median Black OR', value: '0.539',       color: 'text-danger', sub: 'across ten institutions' },
            { label: 'Peer range',               value: '0.456–0.713', color: 'text-warn',   sub: 'largest to smallest gap' },
            { label: 'Hispanic OR (median)',      value: '≈ 1.000',     color: 'text-signal', sub: 'largely at parity' },
            { label: 'Raw approval gap',          value: '19.4 pp',     color: 'text-blue',   sub: 'industry average' },
          ].map(({ label, value, color, sub }) => (
            <div key={label} className="panel-glass rounded-xl p-4 sm:p-5 hover:border-white/10 transition-all">
              <div className={`font-display text-lg sm:text-2xl font-bold ${color}`}>{value}</div>
              <div className="text-bright text-xs mt-1 font-mono">{label}</div>
              <div className="text-ghost text-xs mt-0.5 font-mono">{sub}</div>
            </div>
          ))}
        </div>

        {/* Chart — all bars same color */}
        {chartData.length > 0 && (
          <div className="panel-glass rounded-2xl p-5 sm:p-6 mb-8 sm:mb-10">
            <div className="flex items-center gap-2 mb-2">
              <Shield className="w-4 h-4 text-blue" />
              <span className="text-bright text-sm font-mono">Black-White Adjusted Approval OR by Institution</span>
            </div>
            <p className="text-ghost text-xs font-mono mb-5 sm:mb-6">OR below 1.0 = lower approval odds for Black applicants. Parity = 1.0. All gaps significant at p &lt; 0.001.</p>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={chartData} barSize={24} layout="horizontal">
                <XAxis dataKey="name" tick={{ fill: '#6b7280', fontSize: 9, fontFamily: 'monospace' }} />
                <YAxis domain={[0.3, 1.0]} tick={{ fill: '#6b7280', fontSize: 9, fontFamily: 'monospace' }}
                  tickFormatter={(v: number) => v.toFixed(2)} />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={1.0} stroke="#3a3f4a" strokeDasharray="4 4" label={{ value: 'Parity', fill: '#6b7280', fontSize: 9 }} />
                <Bar dataKey="or" radius={[4, 4, 0, 0]}>
                  {chartData.map((_, i) => (
                    <Cell key={i} fill="#3b82f6" fillOpacity={0.8} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="flex items-center gap-4 mt-4">
              <span className="flex items-center gap-2 text-xs font-mono text-ghost">
                <span className="w-3 h-3 rounded-sm bg-blue/80 inline-block" /> Ten peer institutions, sorted by gap magnitude
              </span>
            </div>
          </div>
        )}

        {/* Table — no Truist highlight */}
        {loading ? (
          <div className="panel-glass rounded-2xl p-12 text-center text-ghost font-mono animate-pulse">Loading...</div>
        ) : (
          <div className="panel-glass rounded-2xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    {['Institution', 'N', 'Black OR', '95% CI', 'Gap'].map((h, i) => (
                      <th key={h} className={`p-3 sm:p-4 text-ghost font-mono text-xs uppercase tracking-widest ${i > 0 ? 'text-right' : 'text-left'}`}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[...data].sort((a, b) => (a.black_OR || 0) - (b.black_OR || 0)).map((row, i) => {
                    const or = row.black_OR || 0
                    const gap = ((1 - or) * 100).toFixed(1)
                    return (
                      <tr key={i} className="border-b border-border/50 hover:bg-white/[0.02] transition-colors">
                        <td className="p-3 sm:p-4 text-bright font-medium">{row.institution || row.name}</td>
                        <td className="p-3 sm:p-4 text-right font-mono text-ghost text-xs">{(row.N || 0).toLocaleString()}</td>
                        <td className="p-3 sm:p-4 text-right font-mono">
                          <span className="text-bright">{or.toFixed(3)}</span>
                        </td>
                        <td className="p-3 sm:p-4 text-right font-mono text-ghost text-xs">
                          [{(row.black_CI_lo || 0).toFixed(3)}, {(row.black_CI_hi || 0).toFixed(3)}]
                        </td>
                        <td className="p-3 sm:p-4 text-right font-mono text-danger text-xs">-{gap}%</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        <div className="mt-8 flex items-start gap-3 text-ghost text-xs font-mono p-4 border border-border/50 rounded-xl">
          <Info className="w-4 h-4 shrink-0 mt-0.5" />
          <span>
            Odds ratios from two-stage MLE logistic regression (statsmodels). Controls: log income, log loan amount,
            DTI midpoint, loan purpose indicators. HMDA does not include credit scores; estimates are upper bounds
            on the true credit-score-adjusted disparity. All gaps significant at p &lt; 0.001.
          </span>
        </div>
      </div>
    </div>
  )
}