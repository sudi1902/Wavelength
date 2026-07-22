// 1080×1920 story card — drawn client-side, downloaded as PNG.
// Orange on ink, one blinking cursor (frozen mid-blink: ON).

import { Tier } from "./tiers";
import { US_ADULT_AVG, GEN_Z_AVG } from "./percentile";

export type CardData = {
  firstName: string;
  hours: number;
  tier: Tier;
  percentile: number;
  rank: number;
  total: number;
  siteUrl: string;
};

const INK = "#141210";
const BLAZE = "#E8720C";
const BURNT = "#C24E00";
const PAPER = "#F2EDE2";
const DIM = "#8F877A";
const BLUE = "#1E2BD8"; // THE BLUE — villain only, one line max

function fmtHours(h: number): string {
  return Number.isInteger(h) ? `${h}` : h.toFixed(2).replace(/0$/, "");
}

export function drawShareCard(
  canvas: HTMLCanvasElement,
  d: CardData,
  monoFamily: string,
  displayFamily: string
): void {
  const W = 1080;
  const H = 1920;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d")!;

  const mono = (px: number, weight = 400) =>
    `${weight} ${px}px ${monoFamily}`;
  const display = (px: number, weight = 900) =>
    `${weight} ${px}px ${displayFamily}`;

  // ground
  ctx.fillStyle = INK;
  ctx.fillRect(0, 0, W, H);

  // halftone field, top-right, fading out
  for (let row = 0; row < 9; row++) {
    for (let col = 0; col < 14; col++) {
      const x = W - 60 - col * 34;
      const y = 178 + row * 34;
      const r = Math.max(0.6, 7 - col * 0.55);
      ctx.globalAlpha = Math.max(0.06, 1 - col * 0.08);
      ctx.fillStyle = BLAZE;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;

  // doc strip
  ctx.fillStyle = DIM;
  ctx.font = mono(22, 500);
  ctx.textBaseline = "top";
  ctx.fillText("BLINK-LDR-001", 64, 64);
  ctx.fillText("V1.0 · FIELD TESTED", 64, 94);
  ctx.textAlign = "right";
  ctx.fillStyle = BLAZE;
  ctx.fillText("STATUS: LOGGED", W - 64, 64);
  ctx.textAlign = "left";

  // rule
  ctx.fillStyle = BLAZE;
  ctx.fillRect(64, 140, W - 128, 3);

  // wordmark + blinking block (frozen ON)
  ctx.fillStyle = PAPER;
  ctx.font = display(150);
  ctx.fillText("BLINK", 56, 176);
  const wm = ctx.measureText("BLINK").width;
  ctx.fillStyle = BLAZE;
  ctx.fillRect(56 + wm + 22, 206, 62, 104);

  // terminal log
  const pad2 = (n: number) => String(n).padStart(2, "0");
  const now = new Date();
  const t0 = `${pad2(now.getHours())}:${pad2(now.getMinutes())}`;
  ctx.font = mono(26);
  const log: Array<[string, string]> = [
    [`[ ${t0}:01 ]`, `SCREENER: ${d.firstName.toUpperCase()}`],
    [`[ ${t0}:02 ]`, `FEED EXPOSURE: ${fmtHours(d.hours)}H / DAY`],
    [`[ ${t0}:02 ]`, `THE BLUE: DETECTED`],
    [`[ ${t0}:03 ]`, `RANK: ${d.rank} OF ${Math.max(d.total, d.rank)}`],
    [`[ ${t0}:03 ]`, `ZERO JUDGMENT. LOGGED.`],
  ];
  let ly = 392;
  for (const [ts, msg] of log) {
    ctx.fillStyle = BLAZE;
    ctx.fillText(ts, 64, ly);
    ctx.fillStyle = msg.startsWith("THE BLUE") ? BLUE : DIM;
    ctx.fillText(msg, 64 + 200, ly);
    ly += 42;
  }

  // giant hours
  ctx.fillStyle = BLAZE;
  ctx.font = display(360);
  const hoursTxt = `${fmtHours(d.hours)}H`;
  ctx.fillText(hoursTxt, 52, 640);

  // percentile statement
  ctx.fillStyle = PAPER;
  ctx.font = display(86, 800);
  ctx.fillText(`MORE FEED THAN`, 60, 1030);
  ctx.fillStyle = BLAZE;
  ctx.fillText(`${d.percentile}% OF SCREENERS`, 60, 1120);

  // tier box
  const tierY = 1268;
  ctx.strokeStyle = BLAZE;
  ctx.lineWidth = 3;
  ctx.strokeRect(64, tierY, W - 128, 150);
  ctx.fillStyle = DIM;
  ctx.font = mono(24, 500);
  ctx.fillText(`TIER ${d.tier.code}`, 92, tierY + 28);
  ctx.fillStyle = BLAZE;
  ctx.font = display(64, 800);
  ctx.fillText(d.tier.name, 92, tierY + 62);
  ctx.fillStyle = PAPER;
  ctx.font = mono(24);
  ctx.fillText(d.tier.line, 92, tierY + 112 + 8);

  // benchmark bars
  const barX = 64;
  const barW = W - 128;
  const maxH = 16;
  const rows: Array<[string, number, string]> = [
    [`YOU`, d.hours, BLAZE],
    [`US ADULT AVG`, US_ADULT_AVG, BURNT],
    [`GEN Z AVG`, GEN_Z_AVG, BURNT],
  ];
  let by = 1490;
  for (const [label, val, color] of rows) {
    ctx.fillStyle = DIM;
    ctx.font = mono(22, 500);
    ctx.fillText(label, barX, by);
    ctx.textAlign = "right";
    ctx.fillStyle = color;
    ctx.fillText(`${fmtHours(val)}H`, barX + barW, by);
    ctx.textAlign = "left";
    ctx.strokeStyle = "#3A332B";
    ctx.lineWidth = 2;
    ctx.strokeRect(barX, by + 34, barW, 26);
    ctx.fillStyle = color;
    ctx.fillRect(
      barX + 2,
      by + 36,
      Math.max(6, (Math.min(val, maxH) / maxH) * (barW - 4)),
      22
    );
    by += 92;
  }

  // serial + pseudo-barcode (deterministic on rank)
  const serial = `NO. ${String(d.rank).padStart(4, "0")}`;
  let bx = 64;
  const bcY = 1800;
  let seed = d.rank * 2654435761 + d.hours * 97;
  for (let i = 0; i < 42 && bx < 420; i++) {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    const w = 2 + (seed % 7);
    if (i % 2 === 0) {
      ctx.fillStyle = PAPER;
      ctx.fillRect(bx, bcY, w, 54);
    }
    bx += w + 2;
  }
  ctx.fillStyle = DIM;
  ctx.font = mono(22, 500);
  ctx.fillText(serial, 64, bcY + 66);

  // bottom: WEAR THE LIGHT + cursor block + url
  ctx.fillStyle = PAPER;
  ctx.font = display(58, 800);
  ctx.textAlign = "right";
  ctx.fillText("WEAR THE LIGHT", W - 64 - 52, bcY - 6);
  ctx.fillStyle = BLAZE;
  ctx.fillRect(W - 64 - 40, bcY + 2, 40, 48);
  ctx.font = mono(24, 500);
  ctx.fillStyle = BLAZE;
  ctx.fillText(
    d.siteUrl.replace(/^https?:\/\//, "").toUpperCase() + "_",
    W - 64,
    bcY + 66
  );
  ctx.textAlign = "left";
}
