'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { BarChart2, ArrowLeft, Info } from 'lucide-react'
import { Logo, BackToTop, ScrollProgress } from '@/components/shared'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function ComplaintsPage() {
  const [summary, setSummary] = useState<any>(null)

  useEffect(() => {
    fetch(`${API}/api/complaints`)
      .then(r => r.json())
      .then(setSummary)
      .catch(() => {})
  }, [])

  const productData = [
    { product: 'Checking/Savings', pre: 1674,  post: 10230, change: 8556  },
    { product: 'Credit Reporting', pre: 332,   post: 5044,  change: 4712  },
    { product: 'Vehicle Loan',     pre: 216,   post: 1596,  change: 1380  },
    { product: 'Debt Collection',  pre: 349,   post: 1322,  change: 973   },
    { product: 'Credit Card',      pre: 513,   post: 1483,  change: 970   },
    { product: 'Mortgage',         pre: 4585,  post: 3143,  change: -1442 },
  ]

  const preGrowth = [
    { name: 'Truist',          rate: 0.98  },
    { name: 'JPMorgan',        rate: 0.42  },
    { name: 'Citibank',        rate: 1.96  },
    { name: 'Wells Fargo',     rate: -0.64 },
    { name: 'Bank of America', rate: -2.14 },
  ]

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
          <span className="text-bright font-mono text-sm">complaints</span>
          <div className="ml-auto">
            <Link href="/"><Logo size={20} /></Link>
          </div>
        </div>
      </nav>

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 pt-24 sm:pt-28 pb-24">
        <div className="mb-10 sm:mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-warn/10">
              <BarChart2 className="w-5 h-5 text-warn" />
            </div>
            <span className="text-xs font-mono uppercase tracking-widest text-ghost">CFPB · Difference-in-Differences</span>
          </div>
          <h1 className="font-display text-3xl sm:text-4xl md:text-5xl font-bold text-white mb-4">
            Complaint Intelligence
          </h1>
          <p className="text-ghost text-base sm:text-lg max-w-2xl">
            Two-way fixed effects DiD comparing Truist to four control institutions after
            the February 2019 BB&T/SunTrust merger announcement. 677,037 matched complaints.
          </p>
        </div>

        {/* DiD stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4 mb-10 sm:mb-12">
          {[
            { label: 'DiD estimate (δ̂)', value: '0.111', color: 'text-warn' },
            { label: 'Implied increase',  value: '+11.8%', color: 'text-warn' },
            { label: 'p-value (HC3)',     value: '0.014',  color: 'text-signal' },
            { label: '95% CI',            value: '[0.022, 0.201]', color: 'text-ghost' },
          ].map(({ label, value, color }) => (
            <div key={label} className="panel-glass rounded-xl p-4 sm:p-5 hover:border-white/10 transition-all">
              <div className={`font-display text-xl sm:text-2xl font-bold ${color}`}>{value}</div>
              <div className="text-ghost text-xs mt-1 font-mono">{label}</div>
            </div>
          ))}
        </div>

        {/* Pre-period growth */}
        <div className="panel-glass rounded-2xl p-5 sm:p-6 mb-5 sm:mb-6">
          <div className="flex flex-wrap items-center gap-2 mb-2">
            <span className="text-bright text-sm font-mono">Pre-Period Quarterly Growth Rates</span>
            <span className="text-xs font-mono text-signal border border-signal/20 bg-signal/5 px-2 py-0.5 rounded-full">Parallel trends support</span>
          </div>
          <p className="text-ghost text-xs font-mono mb-5 sm:mb-6">Pre-period growth rates broadly comparable across institutions — no outlier pattern before the merger.</p>
          <div className="grid grid-cols-3 sm:grid-cols-5 gap-3 sm:gap-3">
            {preGrowth.map(({ name, rate }) => (
              <div key={name} className="text-center panel-glass rounded-xl p-3 hover:border-white/10 transition-all">
                <div className={`font-mono text-sm font-bold mb-1 ${rate > 0 ? 'text-signal' : 'text-ghost'}`}>
                  {rate > 0 ? '+' : ''}{rate.toFixed(2)}%
                </div>
                <div className="text-xs font-mono text-ghost">{name}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Product breakdown */}
        <div className="panel-glass rounded-2xl p-5 sm:p-6 mb-8 sm:mb-10">
          <div className="flex items-center gap-2 mb-5 sm:mb-6">
            <span className="text-bright text-sm font-mono">Post-Merger Complaint Change by Product</span>
          </div>
          <div className="space-y-3 sm:space-y-4">
            {productData.map(({ product, change }) => {
              const isPositive = change > 0
              const pct = Math.abs(change) / 8556 * 100
              return (
                <div key={product} className="flex items-center gap-3 sm:gap-4">
                  <div className="w-28 sm:w-36 text-xs font-mono text-ghost truncate shrink-0">{product}</div>
                  <div className="flex-1 h-5 sm:h-6 bg-border/30 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-1000"
                      style={{
                        width: `${pct}%`,
                        background: isPositive ? 'rgba(245,158,11,0.6)' : 'rgba(0,212,170,0.4)',
                      }}
                    />
                  </div>
                  <div className={`w-16 sm:w-20 text-right font-mono text-xs shrink-0 ${isPositive ? 'text-warn' : 'text-signal'}`}>
                    {isPositive ? '+' : ''}{change.toLocaleString()}
                  </div>
                </div>
              )
            })}
          </div>
          <p className="text-ghost text-xs font-mono mt-5 sm:mt-6">
            Checking/savings accounts account for 73.1% of total incremental positive volume post-merger.
          </p>
        </div>

        {/* Specification */}
        <div className="panel-glass rounded-2xl p-5 sm:p-6 mb-8">
          <div className="text-bright text-sm font-mono mb-4">Model Specification</div>
          <div className="font-mono text-xs text-ghost leading-relaxed">
            <span className="text-soft">log(1 + C</span><span className="text-ghost">ᵢₜ</span>
            <span className="text-soft">) = αᵢ + λₜ + </span>
            <span className="text-signal">δ</span>
            <span className="text-soft"> · (Treatᵢ × Postₜ) + εᵢₜ</span>
            <br /><br />
            <span className="text-ghost">Two-way fixed effects: institution (αᵢ) + quarter (λₜ). Standard errors: HC3-robust.</span>
            <br />
            <span className="text-ghost">Control group: Bank of America, Wells Fargo, JPMorgan Chase, Citibank.</span>
          </div>
        </div>

        <div className="flex items-start gap-3 text-ghost text-xs font-mono p-4 border border-border/50 rounded-xl">
          <Info className="w-4 h-4 shrink-0 mt-0.5" />
          <span>DiD estimates are directionally informative. Control institutions are substantially larger than Truist; log transformation and institution FEs partially address size differences. Maximum observed z-score: 1.50 (no anomalous spikes at z &gt; 2.0).</span>
        </div>
      </div>
    </div>
  )
}