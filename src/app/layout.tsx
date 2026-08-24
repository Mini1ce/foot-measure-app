import type { Metadata, Viewport } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'FootMeasure — AI Foot Measurement',
  description:
    'Estimate foot length and width on-device using a camera and an ID-1 reference card.',
  keywords: ['foot measurement', 'shoe size', 'foot scanner', 'AI measurement', 'mediapipe'],
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  themeColor: '#030712',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="th">
      <head>
        <meta name="mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
      </head>
      <body className="bg-gray-950 text-white antialiased">
        {children}
      </body>
    </html>
  );
}
