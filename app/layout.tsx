import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Road Object Detection',
  description: 'Detect and analyze objects in road scenes',
  generator: 'Next.js',
  applicationName: 'Road Object Detection',
  referrer: 'origin-when-cross-origin',
  keywords: ['Road Object Detection', 'Object Detection', 'Road Scenes'],
  authors: [{ name: 'kasinadhsarma', url: 'https://yourwebsite.com' }],
  colorScheme: 'light dark',
  creator: 'kasinadhsarma',
  publisher: 'kasinadhsarma',
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#000000' },
  ]
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
