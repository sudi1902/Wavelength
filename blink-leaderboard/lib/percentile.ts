// Global percentile = benchmark distribution blended with live entries.
//
// Benchmark: a mixture of two normals anchored on published averages —
// US adults ~7h/day, Gen Z ~9h/day. As the live board grows, its empirical
// distribution takes over from the benchmark.

export const US_ADULT_AVG = 7.0;
export const GEN_Z_AVG = 9.0;

// Abramowitz–Stegun erf approximation (max error ~1.5e-7).
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) *
      t +
      0.254829592) *
      t *
      Math.exp(-ax * ax);
  return sign * y;
}

function normCdf(x: number, mean: number, sd: number): number {
  return 0.5 * (1 + erf((x - mean) / (sd * Math.SQRT2)));
}

export function benchmarkPercentile(hours: number): number {
  // 55% general adult population N(7, 2.4), 45% Gen Z-weighted N(9, 2.6)
  const p =
    0.55 * normCdf(hours, US_ADULT_AVG, 2.4) +
    0.45 * normCdf(hours, GEN_Z_AVG, 2.6);
  return Math.min(99.9, Math.max(0.1, p * 100));
}

export function blendedPercentile(hours: number, liveHours: number[]): number {
  const bench = benchmarkPercentile(hours);
  const n = liveHours.length;
  if (n === 0) return bench;

  let below = 0;
  let equal = 0;
  for (const h of liveHours) {
    if (h < hours) below++;
    else if (h === hours) equal++;
  }
  const empirical = ((below + 0.5 * equal) / n) * 100;

  // Live entries earn weight as the board fills (50/50 at n=40).
  const w = n / (n + 40);
  const blended = bench * (1 - w) + empirical * w;
  return Math.min(99.9, Math.max(0.1, blended));
}
