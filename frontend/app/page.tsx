'use client'

import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { ArrowRight, Shield, Search, AlertTriangle, BarChart2, Menu, X, ChevronDown } from 'lucide-react'
import { Logo, BackToTop, ScrollProgress } from '@/components/shared'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

function useCountUp(target: number, duration = 2200, decimals = 0) {
  const [count, setCount] = useState(0)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => {
      if (!e.isIntersecting) return
      obs.disconnect()
      const start = performance.now()
      const tick = (now: number) => {
        const p = Math.min((now - start) / duration, 1)
        const ease = 1 - Math.pow(1 - p, 4)
        setCount(parseFloat((ease * target).toFixed(decimals)))
        if (p < 1) requestAnimationFrame(tick)
      }
      requestAnimationFrame(tick)
    }, { threshold: 0.2 })
    if (ref.current) obs.observe(ref.current)
    return () => obs.disconnect()
  }, [target, duration, decimals])
  return { count, ref }
}

function useReveal() {
  const ref = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) { setVisible(true); obs.disconnect() }
    }, { threshold: 0.1 })
    if (ref.current) obs.observe(ref.current)
    return () => obs.disconnect()
  }, [])
  return { ref, visible }
}

function StatPill({ label, value, suffix = '', decimals = 0, color = 'signal', prefix = '' }: {
  label: string; value: number; suffix?: string; decimals?: number; color?: string; prefix?: string
}) {
  const { count, ref } = useCountUp(value, 2200, decimals)
  const colorMap: Record<string, string> = {
    signal: 'text-signal', warn: 'text-warn', danger: 'text-danger', blue: 'text-blue',
  }
  return (
    <div ref={ref} className="flex flex-col gap-1 group cursor-default">
      <span className={`font-display text-3xl sm:text-4xl md:text-5xl font-bold tabular-nums transition-all group-hover:scale-105 origin-left ${colorMap[color] || 'text-signal'}`}>
        {prefix}{count.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}{suffix}
      </span>
      <span className="text-ghost text-xs uppercase tracking-widest font-mono">{label}</span>
    </div>
  )
}

function SectionTag({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-[0.2em] text-signal border border-signal/20 bg-signal/5 px-3 py-1.5 rounded-full">
      <span className="w-1.5 h-1.5 rounded-full bg-signal animate-pulse" />
      {children}
    </span>
  )
}

function FindingCard({ icon: Icon, title, value, sub, href, color = 'signal' }: {
  icon: any; title: string; value: string; sub: string; href: string; color?: string
}) {
  const colorMap: Record<string, string> = {
    signal: 'bg-signal/10 text-signal group-hover:bg-signal/20',
    warn:   'bg-warn/10 text-warn group-hover:bg-warn/20',
    danger: 'bg-danger/10 text-danger group-hover:bg-danger/20',
  }
  const { ref, visible } = useReveal()
  return (
    <div ref={ref} className={`transition-all duration-700 ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
      <Link href={href} className="group relative panel-glass rounded-2xl p-6 flex flex-col h-full hover:border-white/10 transition-all duration-300 hover:-translate-y-2 hover:shadow-2xl hover:shadow-black/40 block">
        <div className="flex items-start justify-between mb-5">
          <div className={`p-2.5 rounded-xl transition-all duration-300 ${colorMap[color]}`}>
            <Icon className="w-5 h-5" />
          </div>
          <ArrowRight className="w-4 h-4 text-muted group-hover:text-signal group-hover:translate-x-1 transition-all duration-200" />
        </div>
        <div className="font-display text-3xl sm:text-4xl text-white font-bold mb-2">{value}</div>
        <div className="text-bright text-sm font-medium mb-3">{title}</div>
        <div className="text-ghost text-xs leading-relaxed flex-1">{sub}</div>
        <div className="mt-5 pt-4 border-t border-border/50 flex items-center gap-2 text-xs font-mono text-ghost group-hover:text-signal transition-colors">
          Explore <ArrowRight className="w-3 h-3" />
        </div>
      </Link>
    </div>
  )
}

function DataCard({ n, l }: { n: string; l: string }) {
  const { ref, visible } = useReveal()
  return (
    <div ref={ref} className={`panel-glass rounded-xl p-5 hover:border-white/10 transition-all duration-500 hover:-translate-y-1 ${visible ? 'opacity-100 scale-100' : 'opacity-0 scale-95'}`}>
      <div className="font-display text-2xl text-white font-bold">{n}</div>
      <div className="text-ghost text-xs mt-1 font-mono">{l}</div>
    </div>
  )
}

export default function Home() {
  const [mobileOpen, setMobileOpen] = useState(false)

  const navLinks = [
    { href: '/watchlist', label: 'Watchlist' },
    { href: '/complaints', label: 'Complaints' },
    { href: '/fairlending', label: 'Fair Lending' },
    { href: '/bank', label: 'Bank Lookup' },
  ]

  return (
    <div className="min-h-screen bg-ink noise">
      <ScrollProgress />
      <div className="fixed inset-0 bg-grid-faint bg-grid opacity-100 pointer-events-none" />
      <BackToTop />

      {/* Nav */}
      <nav className="fixed top-0 left-0 right-0 z-50 panel-glass border-b border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2.5 font-display text-lg text-white font-bold tracking-tight">
            <Logo size={26} />
            <span>Byltris</span>
          </Link>
          <div className="hidden md:flex items-center gap-7">
            {navLinks.map(({ href, label }) => (
              <Link key={href} href={href} className="text-ghost hover:text-bright transition-colors text-sm font-mono tracking-wide hover:text-signal">
                {label}
              </Link>
            ))}
          </div>
          <div className="flex items-center gap-3">
            <Link href="/watchlist" className="hidden sm:inline-flex text-xs font-mono bg-signal text-ink px-4 py-2 rounded-full hover:bg-signal/90 transition-colors">
              Dashboard
            </Link>
            <button
              onClick={() => setMobileOpen(v => !v)}
              className="md:hidden p-2 text-ghost hover:text-bright transition-colors"
              aria-label="Toggle menu"
            >
              {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
        {/* Mobile menu */}
        <div className={`md:hidden border-t border-border overflow-hidden transition-all duration-300 ${mobileOpen ? 'max-h-64' : 'max-h-0'}`}>
          <div className="px-6 py-4 flex flex-col gap-4">
            {navLinks.map(({ href, label }) => (
              <Link key={href} href={href} onClick={() => setMobileOpen(false)}
                className="text-ghost hover:text-signal transition-colors text-sm font-mono py-1">
                {label}
              </Link>
            ))}
            <Link href="/watchlist" onClick={() => setMobileOpen(false)}
              className="inline-flex w-fit text-xs font-mono bg-signal text-ink px-4 py-2 rounded-full">
              Dashboard
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="relative min-h-screen flex flex-col justify-center pt-14">
        <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] sm:w-[600px] h-[400px] sm:h-[600px] rounded-full bg-signal/5 blur-[120px] pointer-events-none" />
        <div className="absolute top-1/3 left-1/4 w-[200px] sm:w-[300px] h-[200px] sm:h-[300px] rounded-full bg-blue/5 blur-[100px] pointer-events-none" />

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 py-16 sm:py-24">
          <div className="max-w-4xl">
            <div className="mb-6 sm:mb-8 animate-fade-up" style={{ animationDelay: '0.1s' }}>
              <SectionTag>Consumer Financial Intelligence</SectionTag>
            </div>

            <h1 className="font-display text-4xl sm:text-6xl md:text-7xl lg:text-8xl font-bold leading-[1.05] mb-5 sm:mb-6 animate-fade-up" style={{ animationDelay: '0.2s' }}>
              <span className="text-white">What call reports</span>
              <br />
              <span className="text-gradient">don&apos;t tell you.</span>
            </h1>

            <p className="text-ghost text-base sm:text-lg md:text-xl leading-relaxed mb-8 sm:mb-10 max-w-2xl animate-fade-up" style={{ animationDelay: '0.35s' }}>
              A consumer intelligence platform built on 550,404 bank-quarters of FDIC data,
              14.77 million CFPB complaints, and 4.49 million HMDA mortgage applications.
              Early warning. Fair lending. Complaint intelligence.
            </p>

            <div className="flex flex-wrap gap-3 sm:gap-4 animate-fade-up" style={{ animationDelay: '0.5s' }}>
              <Link href="/watchlist"
                className="inline-flex items-center gap-2 bg-signal text-ink font-mono text-sm px-5 sm:px-6 py-3 rounded-full hover:bg-signal/90 transition-all hover:gap-3 hover:shadow-lg hover:shadow-signal/20">
                Explore Watchlist <ArrowRight className="w-4 h-4" />
              </Link>
              <Link href="/bank"
                className="inline-flex items-center gap-2 border border-border text-soft font-mono text-sm px-5 sm:px-6 py-3 rounded-full hover:border-signal/30 hover:text-signal transition-all">
                <Search className="w-4 h-4" />
                Look up a bank
              </Link>
            </div>
          </div>

          {/* Stats bar */}
          <div className="mt-16 sm:mt-24 grid grid-cols-2 md:grid-cols-4 gap-6 sm:gap-8 border-t border-border pt-10 sm:pt-12 animate-fade-up" style={{ animationDelay: '0.7s' }}>
            <StatPill label="Banks monitored" value={9820} />
            <StatPill label="Early warning AUC" value={0.804} decimals={3} color="blue" />
            <StatPill label="Merger complaint increase" value={11.8} suffix="%" decimals={1} color="warn" />
            <StatPill label="Industry median Black OR" value={0.539} decimals={3} color="danger" />
          </div>
        </div>

        {/* Scroll hint */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-muted animate-scroll-pulse">
          <ChevronDown className="w-5 h-5" />
        </div>
      </section>

      {/* Key Findings */}
      <section className="relative max-w-7xl mx-auto px-4 sm:px-6 py-20 sm:py-32">
        <div className="mb-12 sm:mb-16">
          <SectionTag>Key Findings</SectionTag>
          <h2 className="font-display text-3xl sm:text-4xl md:text-5xl font-bold text-white mt-5 sm:mt-6 mb-4">
            Three lenses. One picture.
          </h2>
          <p className="text-ghost text-base sm:text-lg max-w-2xl">
            Each module answers a distinct question about bank health that no single regulatory dataset answers alone.
          </p>
        </div>

        <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-5 sm:gap-6">
          <FindingCard
            icon={AlertTriangle}
            color="warn"
            title="Early Warning Model"
            value="0.804"
            sub="Out-of-sample AUC on a temporally embargoed holdout spanning Q3 2020 to Q4 2025. 100-fold Precision@50 lift over base rate. Texas Ratio dominates SHAP attribution."
            href="/watchlist"
          />
          <FindingCard
            icon={BarChart2}
            color="signal"
            title="Merger Complaint Surge"
            value="+11.8%"
            sub="DiD estimate (p = 0.014) attributing 11.8% log-complaint increase to the 2019 BB&T/SunTrust merger. 73.1% concentrated in checking and savings accounts."
            href="/complaints"
          />
          <FindingCard
            icon={Shield}
            color="danger"
            title="Fair Lending Gap"
            value="0.539"
            sub="Industry median Black-White adjusted mortgage approval OR across ten institutions. Range: 0.456 to 0.713. Hispanic OR indistinguishable from parity at most institutions."
            href="/fairlending"
          />
        </div>
      </section>

      {/* Methodology */}
      <section className="relative border-y border-border bg-panel/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-16 sm:py-20">
          <div className="grid md:grid-cols-2 gap-10 sm:gap-16 items-center">
            <div>
              <SectionTag>Methodology</SectionTag>
              <h2 className="font-display text-3xl sm:text-4xl font-bold text-white mt-5 sm:mt-6 mb-5 sm:mb-6">
                No proprietary data.<br />No special access.
              </h2>
              <p className="text-ghost leading-relaxed mb-6 text-sm sm:text-base">
                Every result is fully reproducible from public regulatory data — FDIC Statistics on
                Depository Institutions, CFPB Consumer Complaint Database, FFIEC HMDA, and FRED.
                The full pipeline from ingestion to model artifacts is open source.
              </p>
              <div className="flex flex-wrap gap-2 sm:gap-3">
                {['FDIC SDI', 'CFPB Complaints', 'FFIEC HMDA', 'FRED', 'XGBoost', 'statsmodels', 'BERTopic'].map(tag => (
                  <span key={tag} className="text-xs font-mono text-ghost border border-border px-3 py-1.5 rounded-full hover:border-signal/30 hover:text-signal transition-colors cursor-default">
                    {tag}
                  </span>
                ))}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3 sm:gap-4">
              <DataCard n="550,404" l="bank-quarters" />
              <DataCard n="14.77M" l="CFPB complaints" />
              <DataCard n="4.49M" l="HMDA applications" />
              <DataCard n="9,820" l="unique institutions" />
            </div>
          </div>
        </div>
      </section>

      {/* SHAP highlight */}
      <section className="relative max-w-7xl mx-auto px-4 sm:px-6 py-20 sm:py-28">
        <div className="mb-10 sm:mb-12">
          <SectionTag>Model Transparency</SectionTag>
          <h2 className="font-display text-3xl sm:text-4xl font-bold text-white mt-5 sm:mt-6 mb-4">
            SHAP feature attribution.
          </h2>
          <p className="text-ghost max-w-2xl text-sm sm:text-base">
            TreeSHAP values from the XGBoost early warning model. The Texas Ratio dominates — 2.14 mean absolute SHAP, more than twice any other feature.
          </p>
        </div>

        <div className="panel-glass rounded-2xl p-5 sm:p-8 space-y-4">
          {[
            { feature: 'Texas Ratio', shap: 2.142, max: 2.142 },
            { feature: 'Return on Assets', shap: 0.976, max: 2.142 },
            { feature: 'CRE Concentration', shap: 0.808, max: 2.142 },
            { feature: 'Tier 1 Leverage', shap: 0.591, max: 2.142 },
            { feature: 'Loan-to-Deposit Ratio', shap: 0.569, max: 2.142 },
            { feature: 'Texas Ratio (QoQ)', shap: 0.188, max: 2.142 },
            { feature: 'NIM (QoQ)', shap: 0.171, max: 2.142 },
          ].map(({ feature, shap, max }, i) => (
            <ShapRow key={feature} feature={feature} shap={shap} max={max} delay={i * 80} />
          ))}
        </div>
      </section>

      {/* Bank search CTA */}
      <section className="relative max-w-7xl mx-auto px-4 sm:px-6 pb-20 sm:pb-32">
        <div className="panel-glass rounded-2xl sm:rounded-3xl p-8 sm:p-12 md:p-20 text-center border-glow animate-glow-pulse">
          <div className="mb-5 sm:mb-6">
            <SectionTag>Live Lookup</SectionTag>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl md:text-6xl font-bold text-white mb-5 sm:mb-6">
            Search any U.S. bank<br />by CERT number.
          </h2>
          <p className="text-ghost text-base sm:text-lg mb-8 sm:mb-10 max-w-xl mx-auto">
            Full Texas Ratio trajectory, asset history, and distress probability from our XGBoost early warning model.
          </p>
          <Link href="/bank"
            className="inline-flex items-center gap-3 bg-white text-ink font-mono text-sm font-medium px-6 sm:px-8 py-3 sm:py-4 rounded-full hover:bg-bright transition-all hover:gap-4 hover:shadow-xl">
            <Search className="w-4 h-4" />
            Open Bank Lookup
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-10 sm:py-12 flex flex-col sm:flex-row items-center justify-between gap-4">
          <Link href="/" className="flex items-center gap-2 font-display text-white font-bold">
            <Logo size={20} />
            <span>Byltris</span>
          </Link>
          <div className="text-ghost text-xs font-mono text-center">
  Data: FDIC SDI, CFPB, FFIEC HMDA, FRED. For informational purposes only.
</div>
<div className="text-ghost text-xs font-mono text-center">
  Built by{' '}
  <a href="https://linkedin.com/in/abhir-iyer" target="_blank" rel="noopener noreferrer"
    className="text-signal hover:underline transition-colors">
    Abhir Iyer
  </a>
  {' '}· MS Data Science, Indiana University Bloomington
</div>
          <a href="https://github.com/abhir-iyer/byltris" target="_blank" rel="noopener noreferrer"
            className="text-ghost hover:text-signal text-xs font-mono transition-colors">
            GitHub ↗
          </a>
        </div>
      </footer>
    </div>
  )
}

function ShapRow({ feature, shap, max, delay }: { feature: string; shap: number; max: number; delay: number }) {
  const { ref, visible } = useReveal()
  const pct = (shap / max) * 100

  return (
    <div ref={ref} className={`flex items-center gap-3 sm:gap-4 transition-all duration-700 ${visible ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-4'}`}
      style={{ transitionDelay: `${delay}ms` }}>
      <div className="w-32 sm:w-40 text-xs sm:text-sm font-mono text-ghost text-right shrink-0">{feature}</div>
      <div className="flex-1 h-5 sm:h-6 bg-border/30 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full bg-gradient-to-r from-signal to-blue transition-all duration-1000"
          style={{ width: visible ? `${pct}%` : '0%', transitionDelay: `${delay + 200}ms` }}
        />
      </div>
      <div className="w-10 sm:w-12 text-right font-mono text-xs text-signal shrink-0">{shap.toFixed(3)}</div>
    </div>
  )
}