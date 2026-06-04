'use client';
import { useEffect } from 'react';
import { usePathname } from 'next/navigation';
import { initXPixel } from './lib/xpixel';

// Loads the X advertising pixel app-wide, but NOT on chrome-less routes (embed is shown
// inside third-party iframes; print/og are server screenshot targets) — firing an ad
// pixel there would track other people's pages pointlessly.
export default function XPixelInit() {
  const pathname = usePathname();
  useEffect(() => {
    if (
      pathname &&
      (pathname.endsWith('/embed') || pathname.endsWith('/print') || pathname.endsWith('/og'))
    ) {
      return;
    }
    initXPixel();
  }, [pathname]);
  return null;
}
