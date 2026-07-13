# Order Fulfillment SOP — Higgsfield Pipeline

Target: ≤45 min hands-on per Basic order. Credits cost per order: ~77–160.

## Per-order workflow

1. **Intake (5 min).** From the requirements form: product photo(s), vibe, platform. If the photo is bad (blurry, cluttered), politely request a retake: "phone photo, product centered, plain background, near a window."
2. **Hero image pass (5 min, 2 credits).** If the buyer's photo is weak, regenerate a clean hero shot:
   - `generate_image`, model `nano_banana_pro`, with buyer photo as reference + prompt describing a commercial studio setup matching their vibe. Keep label text legible in prompt.
   - If photo is already clean, import it directly: `media_import_url` (or upload).
3. **Video pass (10 min hands-on, ~15 min render, 75–150 credits).**
   - `generate_video`, model `marketing_studio_video`
   - `mode`: pick per vibe — luxury→`product_showcase`, energetic→`hypermotion_oj`, story/warm→`tv_spot`, social-proof→`ugc` or `ugc_selfie_testimonial`, apparel→`ugc_virtual_try_on`, satisfying→`ugc_unboxing_asmr`
   - `resolution`: 720p Basic ($25), 1080p Standard/Premium
   - `aspect_ratio`: 9:16 default; 16:9 recut = second generation or `reframe` tool (cheaper)
   - Prompt formula: `[tone] ad for [PRODUCT NAME]. [Opening visual beat]. [Camera move]. [Second beat]. [On-screen text 'HOOK']. [Music/sound direction].`
4. **QC pass (5 min).** Watch full video. Check: product fidelity vs buyer photo, label text not mangled, no weird artifacts in first 2 seconds (the hook). If bad → regenerate with adjusted prompt (budget 1 retry into pricing).
5. **Optional upscale.** `upscale_video` (topaz, 1080p) if the render needs sharpening — small credit cost, big perceived-quality jump.
6. **Deliver (5 min).** Fiverr delivery message template:

> Here's your ad! 🎬 Delivered: [n] video(s), [ratio], [res], sound included, full commercial rights. Two small notes: [one thing you did deliberately], and [one tip for using it, e.g. "post natively, don't re-upload a watermarked version"]. Your included revision(s) cover tweaks to text, pacing, or tone — just tell me what to change. If you're happy with it, a review helps my new gig enormously.

## Revision policy (protect your time)
- Revision = prompt tweak + regenerate (budgeted). Scope creep ("actually can you add a person") = paid extra, offered politely: "That's a new format — I can add it as a $20 extra."

## Credit economics
- Basic $25: ~77 credits (720p) → healthy margin at any sane credit price
- Standard $75: ~154–302 credits (2× 1080p)
- Premium $150: ~308–604 credits (4× 1080p)
- Current balance: ~745 after portfolio. Top up before accepting a Premium order backlog.

## Templatize (after first 5 orders)
Save winning prompt formulas per category (beauty/food/tech/apparel) in prompts.md in this folder. Each saved formula cuts hands-on time toward 20 min — and the prompt library itself becomes a sellable product later.
