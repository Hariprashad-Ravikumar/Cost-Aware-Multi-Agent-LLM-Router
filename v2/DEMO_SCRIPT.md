# v2 Demo Storyboard: Calibrated Cost-Aware LLM Router

**Target length:** 60-90 seconds. **Throughline:** the live cost-savings meter — the video's job is to make the viewer *watch* the savings happen, not just read a claim.

### Scene 1: Open on the stat card (0:00 - 0:04)
*   **Visual:** The hero card at the top of the live page — "93.9% cheaper. No measurable quality loss." — held on screen just long enough to read.
*   **Voiceover:** "A cost-aware LLM router that's 93.9% cheaper than always using the top model, with no measurable drop in accuracy — verified end-to-end, live."

### Scene 2: The architecture, briefly (0:04 - 0:15)
*   **Visual:** Scroll to the architecture diagram.
*   **Voiceover:** "Instead of asking an LLM to guess how hard a prompt is, a small calibrated model — trained on real labeled outcomes, not a prompt — predicts the probability the cheap tier gets it right, and only escalates when that confidence drops below budget."

### Scene 3: Live routing — easy prompt (0:15 - 0:35)
*   **Visual:** Scroll to "Try it live." Type a simple prompt, hit submit.
*   **Action:** Let the diagram animate — watch the "cheap" node light up as the answer comes back.
*   **Voiceover:** "Watch it happen live. An easy prompt stays on the cheap tier..."

### Scene 4: Live routing — the cost meter moves (0:35 - 0:55)
*   **Visual:** Type 2-3 more prompts in quick succession (mix of easy and the "Hard example" button). Keep the camera on the "Cost savings meter" panel as it updates after each one — actual cost climbing slowly, hypothetical cost climbing faster, the "saved" number growing.
*   **Voiceover:** "...and every request updates this meter in real time — what it actually cost, versus what it would have cost if every prompt went straight to the expensive model. That gap is the whole point."

### Scene 5: Close on the stat card, with scale (0:55 - 1:20)
*   **Visual:** Scroll back to the top, hold on the hero card, specifically the scale-projection line.
*   **Voiceover:** "At real production volume — a million requests a month — that's the difference between $114 and $1,877. Over $21,000 a year, from routing alone. All code, real evaluation data, and the honest limitations are on GitHub."
*   **Action:** End frame: the stat card, GitHub URL, live demo URL.

### Notes for recording
- Do the "Try it live" section as one continuous take if possible — the meter's numbers updating live is the actual proof, and it reads as more credible unedited.
- Have the "Hard example" prompt ready to paste in case a genuinely hard prompt is needed to show the diagram highlighting the "mid" or "capable" node instead of "cheap" — don't force an escalation that doesn't happen naturally.
- If recording against the live Cloud Run deployment rather than localhost, do a warm-up request first off-camera — the first request after an idle period can take ~40s (cold start), which would kill the pacing on camera.
