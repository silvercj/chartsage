import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import PostHogInit from './PostHogInit'
import SessionWatcher from './components/SessionWatcher'
import AppHeader from './components/AppHeader'
import { CreditsProvider } from './lib/useCredits'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'ChartSage - AI-Powered Data Visualization',
  description: 'Turn any Excel file into a beautiful, interactive dashboard with AI-generated insights in seconds.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <PostHogInit />
        <SessionWatcher />
        <CreditsProvider>
          <AppHeader />
          <main className="min-h-screen bg-gray-50">{children}</main>
        </CreditsProvider>
      </body>
    </html>
  )
}
