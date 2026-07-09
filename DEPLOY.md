# Deploying Wavelength publicly

This guide takes the app from your Mac to a public URL anyone can use.
Every visitor gets their own anonymous library (a browser cookie); nobody
can see anyone else's sounds, submit file paths, or touch the server's
filesystem — that's `--public` mode, and it's covered by the test suite
(`tests/test_public_mode.py`).

**Never expose local mode to the internet.** `wavelength serve` without
`--public` trusts its user with file paths by design; the CLI refuses to
bind beyond localhost without the flag for exactly this reason.

## What you need

| Thing | Cost | Why |
|---|---|---|
| Hetzner Cloud account | — | hosts the server |
| CPX41 (8 vCPU / 16 GB) or CCX23 (4 dedicated / 16 GB) | ~€25–29/mo | CPU inference needs the RAM |
| A domain (optional but recommended) | ~$10/yr | wavelength.yourdomain.com |
| Cloudflare free account (optional) | $0 | TLS + basic abuse shielding |
| Freesound API key (optional) | $0 | enables "find cleaner" for visitors |

Server-side inference runs on CPU: a 30–60 s clip takes roughly 1–3
minutes, one at a time (the queue is visible to users). The next
optimization — serverless GPU via Modal (~$6/mo at low volume, seconds per
clip) — swaps in behind the same interface later; see PLAN.md Phase 4b.

## Steps

### 1. Create the server

Hetzner Cloud console → Add Server → location near you → image **Docker CE**
(under Apps) → type **CPX41** → add your SSH key → create. Note the IP.

### 2. Install Wavelength

SSH in and run:

```bash
git clone https://github.com/sudi1902/Wavelength.git
cd Wavelength
git checkout claude/sound-effect-extractor-plan-45em07
export WAVELENGTH_FREESOUND_KEY="your-key-here"   # optional
docker compose up -d --build
```

First boot downloads the models into `./data` (~10 minutes, one-time —
watch with `docker compose logs -f`). When the log says
`Wavelength UI (PUBLIC multi-user)`, it's live at `http://SERVER_IP:8317`.

### 3. Put Cloudflare in front (TLS + a real domain)

1. Add your domain to Cloudflare (free plan), set an **A record**:
   `wavelength → SERVER_IP`, proxy status **on** (orange cloud).
2. SSL/TLS mode: **Flexible** (or set up an origin cert for Full).
3. Optionally firewall the server so port 8317 only accepts Cloudflare's
   IP ranges (Hetzner Cloud Firewall).

Your app is now at `https://wavelength.yourdomain.com`.

### 4. Upgrades

```bash
cd Wavelength && git pull && docker compose up -d --build
```

Models and all visitor libraries live in `./data` and survive rebuilds.

## Built-in public-mode protections

- Anonymous session cookie per browser; all effects/jobs scoped to it;
  cross-session access returns 404 (indistinguishable from nonexistent)
- Extraction accepts **URLs only** — file paths rejected outright
- Watch-folder API disabled (403)
- Quotas: 10 extractions/browser/day, 30/IP/day (constants in
  `server/app.py`)
- Source archiving off by default on servers (disk growth)
- One extraction at a time (single worker) — a queue, not a fork bomb

## Operational notes

- **Disk**: visitor WAVs accumulate in `./data/library`. At validation
  scale this is megabytes/day; add a cron purge of old anonymous sessions
  before any real launch (planned with accounts in Phase 4b).
- **Legal posture**: this service downloads platform videos on visitors'
  behalf. Fine for a small validation run; revisit terms before promoting
  it widely (PLAN.md Phase 4 notes).
- **Backups**: `./data` is the whole state; snapshot it or rsync it.
