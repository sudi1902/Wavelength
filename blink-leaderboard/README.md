# BLINK — SCREEN TIME LEADERBOARD v1.0

`BLINK-LDR-001 · STATUS: READY TO DEPLOY`

A viral screen-time leaderboard. Screeners log yesterday's hours + first name +
email → get a global percentile (benchmarked against US adult ~7h and Gen Z ~9h
averages, blended with live entries), a lore tier, and a downloadable
1080×1920 share card for stories.

- **Public board**: first name + hours + tier. Nothing else.
- **Emails**: stored privately as the Day One Deck waitlist. Never exposed —
  the table has RLS enabled with zero policies, and all reads/writes go through
  server-side API routes using the service role key.
- **Zero judgment**: per brand protocol, nothing on this page scolds where you
  look.

## Stack

Next.js (App Router, TypeScript) · Supabase (Postgres) · Vercel.
IBM Plex Mono + Big Shoulders Display via `next/font`. No CSS framework.

## Setup

### 1. Supabase

1. Create a project at [supabase.com](https://supabase.com).
2. Run `supabase/migrations/0001_blink_entries.sql` in the SQL editor
   (or `supabase db push` with the CLI).
3. Grab from **Project Settings → API**:
   - Project URL → `NEXT_PUBLIC_SUPABASE_URL`
   - `service_role` secret key → `SUPABASE_SERVICE_ROLE_KEY`

### 2. Local dev

```bash
cd blink-leaderboard
cp .env.example .env.local   # fill in the values
npm install
npm run dev
```

### 3. Vercel

1. Import the repo on [vercel.com](https://vercel.com/new); set
   **Root Directory** to `blink-leaderboard`.
2. Add the three env vars from `.env.example`
   (`SUPABASE_SERVICE_ROLE_KEY` must NOT be prefixed `NEXT_PUBLIC_`).
3. Deploy. Set `NEXT_PUBLIC_SITE_URL` to the final domain and redeploy so the
   share card prints the right URL.

## How the percentile works

`lib/percentile.ts` models the benchmark as a mixture of two normal
distributions — 55% general US adults N(7h, 2.4) and 45% Gen Z N(9h, 2.6) —
and blends that with the empirical percentile from live board entries. Live
data earns weight as the board grows (50/50 at 40 entries), so the number is
believable on day one and self-correcting at scale.

## Tiers

| Code | Tier               | Hours  |
| ---- | ------------------ | ------ |
| T-01 | ANALOG GHOST       | 0–2    |
| T-02 | SOFT GLOW          | 2–4    |
| T-03 | STEADY STREAM      | 4–6    |
| T-04 | FEED NATIVE        | 6–8    |
| T-05 | BLUE HOUR OPERATOR | 8–10   |
| T-06 | TERMINAL VELOCITY  | 10–13  |
| T-07 | THE GOLDFISH       | 13+    |

One entry per email — re-submitting updates your hours (and the waitlist stays
deduplicated).

## Brand notes

Follows `BLINK-BOS-001`: orange leads every composition, deadpan status lines,
one blinking cursor, halftone fields, THE BLUE quarantined to a single
"DETECTED" status line (≤10%), versioned everything, no moralizing about
screen time. Colors use the requested `#E8720C` on `#141210` with the spec's
PAPER `#F2EDE2` and BURNT `#C24E00` as support.
