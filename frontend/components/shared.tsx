'use client'
import { useEffect, useState } from 'react'
import Link from 'next/link'
import { ArrowUp } from 'lucide-react'

import Image from 'next/image'

export function Logo({ size = 28 }: { size?: number }) {
  return (
    <Image
      src="/favicon-512.png"
      alt="Byltris"
      width={size}
      height={size}
      className="rounded-md"
      priority
    />
  )
}

export function BackToTop() {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const onScroll = () => setVisible(window.scrollY > 400)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  const scrollTop = () => window.scrollTo({ top: 0, behavior: 'smooth' })

  return (
    <button
      onClick={scrollTop}
      aria-label="Back to top"
      className={`fixed bottom-6 right-6 z-50 p-3 rounded-full bg-signal text-ink shadow-lg shadow-signal/20 transition-all duration-300 hover:scale-110 hover:shadow-signal/40 ${
        visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4 pointer-events-none'
      }`}
    >
      <ArrowUp className="w-5 h-5" />
    </button>
  )
}

export function ScrollProgress() {
  const [pct, setPct] = useState(0)
  useEffect(() => {
    const onScroll = () => {
      const el = document.documentElement
      const scrolled = el.scrollTop
      const total = el.scrollHeight - el.clientHeight
      setPct(total > 0 ? (scrolled / total) * 100 : 0)
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])
  return (
    <div className="fixed top-0 left-0 right-0 z-[60] h-0.5 bg-transparent">
      <div
        className="h-full bg-gradient-to-r from-signal to-blue transition-all duration-100"
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}