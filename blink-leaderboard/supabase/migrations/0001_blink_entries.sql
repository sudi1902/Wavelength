-- BLINK — SCREEN TIME LEADERBOARD — v1.0
-- One table. Emails private, board public via server routes only.

create table if not exists public.blink_entries (
  id         uuid primary key default gen_random_uuid(),
  first_name text not null check (char_length(first_name) between 1 and 24),
  email      text not null,
  hours      numeric(4, 2) not null check (hours >= 0 and hours <= 24),
  tier       text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- One entry per screener (emails are lowercased server-side before insert).
-- Re-submitting updates your hours.
alter table public.blink_entries
  add constraint blink_entries_email_key unique (email);

create index if not exists blink_entries_hours_idx
  on public.blink_entries (hours desc);

create index if not exists blink_entries_created_idx
  on public.blink_entries (created_at desc);

-- RLS on, zero policies: anon and authenticated keys can read NOTHING.
-- All access goes through the service role in Next.js API routes, which
-- only ever return first_name + hours + tier to the public.
alter table public.blink_entries enable row level security;
