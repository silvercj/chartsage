import type { Metadata } from 'next'
import { GeistSans, GeistMono, fraunces } from './lib/fonts'
import PostHogInit from './PostHogInit'
import SessionWatcher from './components/SessionWatcher'
import AppHeader from './components/AppHeader'
import { CreditsProvider } from './lib/useCredits'
import './globals.css'

export const metadata: Metadata = {
  title: 'ChartSage - AI-Powered Data Visualization',
  description: 'Turn any spreadsheet into a beautiful, interactive report with AI-generated insights in seconds.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable} ${fraunces.variable}`}>
      <body className="font-sans antialiased">
        <div className="h-[3px] bg-[linear-gradient(90deg,rgb(var(--accent))_0%,rgb(var(--accent))_62%,rgb(var(--ember))_62%,rgb(var(--ember))_100%)]" />
        <PostHogInit />
        <SessionWatcher />
        <CreditsProvider>
          <AppHeader />
          <main className="min-h-screen bg-canvas text-ink">{children}</main>
        </CreditsProvider>
      </body>
    </html>
  )
}
