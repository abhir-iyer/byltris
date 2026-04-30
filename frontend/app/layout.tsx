import type { Metadata } from 'next'
import { DM_Sans, DM_Mono, Playfair_Display } from 'next/font/google'
import './globals.css'

const dmSans = DM_Sans({ subsets: ['latin'], variable: '--font-body', display: 'swap' })
const playfair = Playfair_Display({ subsets: ['latin'], variable: '--font-display', display: 'swap' })
const dmMono = DM_Mono({ subsets: ['latin'], weight: ['400','500'], variable: '--font-mono', display: 'swap' })

export const metadata: Metadata = {
  title: 'Byltris — Bank Health Intelligence',
  description: 'Early warning models, fair lending analysis, and consumer complaint intelligence for U.S. bank supervision.',
  metadataBase: new URL('https://byltris.fyi'),
  icons: {
  icon: [
    { url: '/favicon-16.png', sizes: '16x16', type: 'image/png' },
    { url: '/favicon-32.png', sizes: '32x32', type: 'image/png' },
  ],
  apple: '/apple-touch-icon.png',
},
  openGraph: {
    title: 'Byltris — Bank Health Intelligence',
    description: 'Early warning models, fair lending analysis, and consumer complaint intelligence.',
    type: 'website',
  },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${dmSans.variable} ${playfair.variable} ${dmMono.variable}`}>
      <body className="bg-ink text-bright antialiased font-body">
        {children}
      </body>
    </html>
  )
}