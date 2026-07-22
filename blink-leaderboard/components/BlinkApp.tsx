"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Tier } from "@/lib/tiers";
import { US_ADULT_AVG, GEN_Z_AVG } from "@/lib/percentile";
import { drawShareCard } from "@/lib/shareCard";

type Result = {
  firstName: string;
  hours: number;
  tier: Tier;
  percentile: number;
  rank: number;
  total: number;
};

type BoardRow = { first_name: string; hours: number; tier: string };
type Board = { top: BoardRow[]; recent: BoardRow[]; total: number };

function fmtHours(h: number): string {
  return Number.isInteger(h) ? `${h}` : h.toFixed(2).replace(/0$/, "");
}

function pad2(n: number) {
  return String(n).padStart(2, "0");
}

function ts(offset = 0): string {
  const d = new Date(Date.now() + offset * 1000);
  return `[ ${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())} ]`;
}

export default function BlinkApp() {
  const [firstName, setFirstName] = useState("");
  const [email, setEmail] = useState("");
  const [hours, setHours] = useState(7);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [board, setBoard] = useState<Board | null>(null);
  const [boardOffline, setBoardOffline] = useState(false);
  const [copied, setCopied] = useState(false);
  const resultRef = useRef<HTMLDivElement>(null);

  const loadBoard = useCallback(async () => {
    try {
      const res = await fetch("/api/leaderboard");
      if (!res.ok) throw new Error();
      setBoard(await res.json());
      setBoardOffline(false);
    } catch {
      setBoardOffline(true);
    }
  }, []);

  useEffect(() => {
    loadBoard();
  }, [loadBoard]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const res = await fetch("/api/entries", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ firstName, email, hours }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "INSTALL FAILED. RETRY.");
      setResult(data);
      loadBoard();
      setTimeout(
        () => resultRef.current?.scrollIntoView({ behavior: "smooth" }),
        60
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "INSTALL FAILED. RETRY.");
    } finally {
      setSubmitting(false);
    }
  }

  async function downloadCard() {
    if (!result) return;
    await document.fonts.ready;
    const styles = getComputedStyle(document.body);
    const monoFamily =
      styles.getPropertyValue("--font-mono").trim() || "monospace";
    const displayFamily =
      styles.getPropertyValue("--font-display").trim() || monoFamily;
    const siteUrl =
      process.env.NEXT_PUBLIC_SITE_URL ?? window.location.origin;

    const canvas = document.createElement("canvas");
    drawShareCard(canvas, { ...result, siteUrl }, monoFamily, displayFamily);
    const a = document.createElement("a");
    a.download = `BLINK-${result.tier.name.replace(/\s+/g, "-")}-${fmtHours(result.hours)}H.png`;
    a.href = canvas.toDataURL("image/png");
    a.click();
  }

  async function copyLink() {
    try {
      await navigator.clipboard.writeText(window.location.origin);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* clipboard unavailable — ignore */
    }
  }

  const maxBar = 16;

  return (
    <div className="wrap">
      <div className="doc-strip">
        <span>
          DOCUMENT <b>BLINK-LDR-001</b>
        </span>
        <span>
          VERSION <b>LEADERBOARD v1.0</b>
        </span>
        <span>
          STATUS <b>LIVE · FIELD TESTED</b>
        </span>
        <span>
          ENTRIES <b>{board ? String(board.total).padStart(4, "0") : "----"}</b>
        </span>
      </div>

      <div className="masthead">
        <h1>BLINK</h1>
        <div className="block" aria-hidden />
      </div>
      <div className="tagline">SUNSCREEN FOR THE FEED_ · SCREEN TIME LEADERBOARD</div>

      <section className="hero">
        <h2>
          HOW LONG DO YOU <em>WEAR THE LIGHT?</em>
        </h2>
        <div className="termlines">
          <div>
            <span className="t">{ts(0)}</span> US ADULT AVERAGE: {US_ADULT_AVG}H / DAY
          </div>
          <div>
            <span className="t">{ts(1)}</span> GEN Z AVERAGE: {GEN_Z_AVG}H / DAY
          </div>
          <div>
            <span className="t">{ts(1)}</span>{" "}
            <span className="blue">THE BLUE: DETECTED</span>
          </div>
          <div>
            <span className="t">{ts(2)}</span> LOG YOUR HOURS. CLAIM YOUR TIER. ZERO
            JUDGMENT.
            <span className="cursor" aria-hidden />
          </div>
        </div>
      </section>

      {/* ---- ENTRY FORM ---- */}
      <section className="panel" id="log">
        <div className="panel-head">
          <span>01 — LOG YOUR EXPOSURE</span>
          <span>INPUT REQUIRED</span>
        </div>
        <form className="panel-body" onSubmit={submit}>
          <div className="field">
            <label htmlFor="fn">FIRST NAME — GOES ON THE BOARD</label>
            <input
              id="fn"
              type="text"
              maxLength={24}
              required
              value={firstName}
              onChange={(e) => setFirstName(e.target.value)}
              placeholder="ANA"
              autoComplete="given-name"
            />
          </div>
          <div className="field">
            <label htmlFor="em">EMAIL — STAYS PRIVATE. WAITLIST ONLY.</label>
            <input
              id="em"
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="ANA@FEED.NET"
              autoComplete="email"
            />
          </div>
          <div className="field">
            <label htmlFor="hr">SCREEN TIME YESTERDAY — HOURS</label>
            <div className="hours-row">
              <input
                id="hr"
                type="range"
                min={0}
                max={16}
                step={0.25}
                value={Math.min(hours, 16)}
                onChange={(e) => setHours(Number(e.target.value))}
              />
              <div className="hours-readout">
                {fmtHours(hours)}
                <small>H</small>
              </div>
            </div>
          </div>
          <button className="install" type="submit" disabled={submitting}>
            {submitting ? "INSTALLING…" : "INSTALL ENTRY_"}
          </button>
          {error && <div className="error-line">✕ {error}</div>}
          <div className="fineprint">
            PUBLIC BOARD SHOWS FIRST NAME + HOURS + TIER ONLY. EMAIL IS STORED
            PRIVATELY AS THE DAY ONE DECK WAITLIST. NO SPAM. NO DYES. NO
            MORALIZING.
          </div>
        </form>
      </section>

      {/* ---- RESULT ---- */}
      {result && (
        <section className="panel" ref={resultRef}>
          <div className="panel-head">
            <span>02 — READOUT</span>
            <span>
              TIER {result.tier.code} · RANK {result.rank}/
              {Math.max(result.total, result.rank)}
            </span>
          </div>
          <div className="panel-body">
            <div className="result-log">
              <div>
                <span className="t">{ts(0)}</span> SCREENER:{" "}
                {result.firstName.toUpperCase()}
              </div>
              <div>
                <span className="t">{ts(0)}</span> FEED EXPOSURE:{" "}
                {fmtHours(result.hours)}H / DAY
              </div>
              <div>
                <span className="t">{ts(1)}</span> ENTRY INSTALLED. ZERO
                JUDGMENT.
              </div>
            </div>

            <div className="rank-giant">
              MORE FEED THAN {result.percentile}%
            </div>
            <div className="rank-sub">
              OF SCREENERS, GLOBALLY. BENCHMARKED + LIVE.
            </div>

            <div className="tier-chip">
              {result.tier.code} / {result.tier.name} — {result.tier.line}
            </div>

            <div className="bars">
              {[
                { label: "YOU", v: result.hours, you: true },
                { label: "US ADULT AVG", v: US_ADULT_AVG, you: false },
                { label: "GEN Z AVG", v: GEN_Z_AVG, you: false },
              ].map((b) => (
                <div className="bar-row" key={b.label}>
                  <div className="bar-label">
                    <span>{b.label}</span>
                    <span>{fmtHours(b.v)}H</span>
                  </div>
                  <div className="bar-track">
                    <div
                      className={`bar-fill${b.you ? " you" : ""}`}
                      style={{
                        width: `${Math.min(100, (b.v / maxBar) * 100)}%`,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>

            <div className="actions">
              <button className="btn-primary" onClick={downloadCard}>
                ↓ DOWNLOAD CARD — 1080×1920_
              </button>
              <button className="btn-ghost" onClick={copyLink}>
                {copied ? "COPIED_" : "COPY LINK"}
              </button>
            </div>
          </div>
        </section>
      )}

      {/* ---- RECENT TICKER ---- */}
      {board && board.recent.length > 0 && (
        <div className="ticker" aria-hidden>
          <div className="ticker-inner">
            {[...board.recent, ...board.recent].map((r, i) => (
              <span key={i}>
                <b>{r.first_name}</b> · {fmtHours(Number(r.hours))}H · {r.tier}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* ---- LEADERBOARD ---- */}
      <section className="panel">
        <div className="panel-head">
          <span>03 — THE BOARD</span>
          <span>FIRST NAME + HOURS + TIER. NOTHING ELSE.</span>
        </div>
        <div className="panel-body">
          {boardOffline ? (
            <div className="board-empty">
              ✕ BOARD OFFLINE — SUPABASE NOT CONNECTED. SEE README.
            </div>
          ) : !board ? (
            <div className="board-empty">
              LOADING BOARD<span className="cursor" aria-hidden />
            </div>
          ) : board.top.length === 0 ? (
            <div className="board-empty">
              BOARD EMPTY. FRAME 0001 IS YOURS.
              <span className="cursor" aria-hidden />
            </div>
          ) : (
            <table className="board">
              <thead>
                <tr>
                  <th>NO.</th>
                  <th>SCREENER</th>
                  <th>HOURS</th>
                  <th>TIER</th>
                </tr>
              </thead>
              <tbody>
                {board.top.map((r, i) => (
                  <tr key={i} className={i < 3 ? "top3" : ""}>
                    <td className="rank">{String(i + 1).padStart(4, "0")}</td>
                    <td className="name">{r.first_name.toUpperCase()}</td>
                    <td className="hours">{fmtHours(Number(r.hours))}H</td>
                    <td className="tier">{r.tier}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </section>

      <div className="halftone" aria-hidden />

      <footer>
        <div>
          <div className="wear">
            WEAR THE <i>LIGHT</i>
            <span className="cursor" aria-hidden />
          </div>
          <div style={{ marginTop: 8 }}>
            WE DON&apos;T CARE WHERE YOU LOOK — WE CARE THAT YOU CAN.
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div>BLINK — OPTIC STACK v1.0</div>
          <div>NO DYES. NO JUDGMENT. NO SPAM.</div>
          <div>© {new Date().getFullYear()} BLINK</div>
        </div>
      </footer>
    </div>
  );
}
