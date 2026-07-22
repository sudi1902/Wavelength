// LORE TIERS — deadpan status lines. Zero judgment: higher hours is not a
// failure state, it is more feed. BLINK has no opinion on where you look.

export type Tier = {
  code: string;
  name: string;
  line: string;
  min: number; // inclusive
  max: number; // exclusive
};

export const TIERS: Tier[] = [
  { code: "T-01", name: "ANALOG GHOST",       line: "SIGNAL FAINT. THE FEED WAITS.",             min: 0,  max: 2 },
  { code: "T-02", name: "SOFT GLOW",          line: "AMBIENT EXPOSURE. PULSE STEADY.",           min: 2,  max: 4 },
  { code: "T-03", name: "STEADY STREAM",      line: "RECEPTION NOMINAL. EYES ONLINE.",           min: 4,  max: 6 },
  { code: "T-04", name: "FEED NATIVE",        line: "US ADULT BASELINE CLEARED.",                min: 6,  max: 8 },
  { code: "T-05", name: "BLUE HOUR OPERATOR", line: "GEN Z BASELINE CLEARED. DEFENSE ADVISED.",  min: 8,  max: 10 },
  { code: "T-06", name: "TERMINAL VELOCITY",  line: "SUSTAINED MAXIMUM FEED. RESPECT.",          min: 10, max: 13 },
  { code: "T-07", name: "THE GOLDFISH",       line: "ATTENTION, KEPT ALIVE.",                    min: 13, max: 25 },
];

export function tierFor(hours: number): Tier {
  return (
    TIERS.find((t) => hours >= t.min && hours < t.max) ?? TIERS[TIERS.length - 1]
  );
}
