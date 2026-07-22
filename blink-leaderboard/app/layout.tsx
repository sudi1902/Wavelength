import type { Metadata } from "next";
import { IBM_Plex_Mono, Big_Shoulders } from "next/font/google";
import "./globals.css";

const plexMono = IBM_Plex_Mono({
  weight: ["400", "500", "600", "700"],
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

const bigShoulders = Big_Shoulders({
  weight: ["700", "800", "900"],
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
});

const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL ?? "https://blink-leaderboard.vercel.app";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: "BLINK — SCREEN TIME LEADERBOARD",
  description:
    "Log your daily screen time. Get your global percentile, your tier, your card. Zero judgment — we don't care where you look, we care that you can.",
  openGraph: {
    title: "BLINK — SCREEN TIME LEADERBOARD v1.0",
    description:
      "How long do you wear the light? Log your hours. Claim your tier. Sunscreen for the feed.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${plexMono.variable} ${bigShoulders.variable}`}>
      <body>{children}</body>
    </html>
  );
}
