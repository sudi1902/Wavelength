import { NextResponse } from "next/server";
import { supabaseAdmin } from "@/lib/supabase";

export const dynamic = "force-dynamic";

// Public board: first name + hours + tier ONLY. Emails never leave the server.
export async function GET() {
  const db = supabaseAdmin();
  if (!db) {
    return NextResponse.json(
      { error: "BOARD OFFLINE — SUPABASE NOT CONFIGURED" },
      { status: 503 }
    );
  }

  const [top, recent, count] = await Promise.all([
    db
      .from("blink_entries")
      .select("first_name, hours, tier")
      .order("hours", { ascending: false })
      .order("updated_at", { ascending: true })
      .limit(100),
    db
      .from("blink_entries")
      .select("first_name, hours, tier")
      .order("updated_at", { ascending: false })
      .limit(12),
    db.from("blink_entries").select("id", { count: "exact", head: true }),
  ]);

  if (top.error || recent.error) {
    return NextResponse.json({ error: "READ FAILED" }, { status: 500 });
  }

  return NextResponse.json({
    top: top.data ?? [],
    recent: recent.data ?? [],
    total: count.count ?? 0,
  });
}
