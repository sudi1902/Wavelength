import { NextRequest, NextResponse } from "next/server";
import { supabaseAdmin } from "@/lib/supabase";
import { tierFor } from "@/lib/tiers";
import { blendedPercentile } from "@/lib/percentile";

export const dynamic = "force-dynamic";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

export async function POST(req: NextRequest) {
  const db = supabaseAdmin();
  if (!db) {
    return NextResponse.json(
      { error: "BOARD OFFLINE — SUPABASE NOT CONFIGURED" },
      { status: 503 }
    );
  }

  let body: { firstName?: unknown; email?: unknown; hours?: unknown };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "MALFORMED PAYLOAD" }, { status: 400 });
  }

  const firstName = String(body.firstName ?? "").trim().slice(0, 24);
  const email = String(body.email ?? "").trim().toLowerCase();
  const hours = Math.round(Number(body.hours) * 4) / 4; // quarter-hour steps

  if (firstName.length < 1) {
    return NextResponse.json({ error: "FIRST NAME REQUIRED" }, { status: 400 });
  }
  if (!EMAIL_RE.test(email) || email.length > 254) {
    return NextResponse.json({ error: "EMAIL INVALID" }, { status: 400 });
  }
  if (!Number.isFinite(hours) || hours < 0 || hours > 24) {
    return NextResponse.json(
      { error: "HOURS MUST BE 0–24" },
      { status: 400 }
    );
  }

  const tier = tierFor(hours);

  // One entry per email — re-submitting updates your hours.
  const { error: upsertError } = await db
    .from("blink_entries")
    .upsert(
      {
        first_name: firstName,
        email,
        hours,
        tier: tier.name,
        updated_at: new Date().toISOString(),
      },
      { onConflict: "email" }
    );

  if (upsertError) {
    // onConflict against an expression index isn't supported everywhere;
    // fall back to manual update-or-insert.
    const { data: existing } = await db
      .from("blink_entries")
      .select("id")
      .eq("email", email)
      .maybeSingle();

    const write = existing
      ? db
          .from("blink_entries")
          .update({
            first_name: firstName,
            hours,
            tier: tier.name,
            updated_at: new Date().toISOString(),
          })
          .eq("id", existing.id)
      : db.from("blink_entries").insert({
          first_name: firstName,
          email,
          hours,
          tier: tier.name,
        });

    const { error } = await write;
    if (error) {
      return NextResponse.json({ error: "WRITE FAILED" }, { status: 500 });
    }
  }

  // Percentile against benchmark + live board.
  const { data: rows } = await db
    .from("blink_entries")
    .select("hours")
    .order("hours", { ascending: false })
    .limit(5000);

  const liveHours = (rows ?? []).map((r) => Number(r.hours));
  const percentile = blendedPercentile(hours, liveHours);
  const total = liveHours.length;
  const rank = liveHours.filter((h) => h > hours).length + 1;

  return NextResponse.json({
    firstName,
    hours,
    tier,
    percentile: Math.round(percentile),
    rank,
    total,
  });
}
