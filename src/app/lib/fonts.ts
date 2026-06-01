import { GeistSans } from 'geist/font/sans';
import { GeistMono } from 'geist/font/mono';
import { Fraunces } from 'next/font/google';

// Display serif — variable, optical sizing + italic.
export const fraunces = Fraunces({
  subsets: ['latin'],
  axes: ['opsz'],
  style: ['normal', 'italic'],
  variable: '--font-fraunces',
  display: 'swap',
});

// Geist exposes fixed CSS vars: --font-geist-sans / --font-geist-mono
export { GeistSans, GeistMono };
