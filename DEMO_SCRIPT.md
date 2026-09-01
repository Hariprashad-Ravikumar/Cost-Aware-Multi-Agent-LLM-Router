# RouteDemo: Video Demo Storyboard

**Target Length:** 60-90 seconds.
**Goal:** Watch one real prompt flow through the live n8n canvas, then land on the real cost/accuracy/latency numbers.

### Scene 1: Introduction (0:00 - 0:12)
*   **Visual:** README's architecture Mermaid diagram.
*   **Voiceover:** "Using the most capable LLM for every prompt is expensive. This is a cost-aware router built in n8n that classifies each prompt's difficulty, then sends it to a cheap model or a capable one — benchmarked with real, logged API metrics."

### Scene 2: Live trace — an easy prompt (0:12 - 0:35)
*   **Visual:** n8n editor, `RouteDemo` workflow open. Click "Execute workflow" (or fire a webhook call from a second terminal) with an easy GSM8K-style prompt.
*   **Action:** Let n8n's live execution animation highlight each node in sequence: Webhook → Difficulty Classifier (Groq `openai/gpt-oss-20b`) → Switch → Cheap Model (Groq `openai/gpt-oss-120b`) → Compute Metrics → Respond.
*   **Voiceover:** "The classifier rates this a 1 — trivial — so it routes to the cheap Groq tier and answers in under two seconds."
*   **Action:** Click the Switch node to briefly show the score output.

### Scene 3: Live trace — a hard prompt (0:35 - 0:55)
*   **Visual:** Fire a second webhook call with a genuinely hard prompt (e.g. an MMLU question needing specialized knowledge, or an open-research-style question).
*   **Action:** Watch the canvas light up the *other* branch this time: Switch → Capable Model (Gemini `gemini-3.1-flash-lite`).
*   **Voiceover:** "A harder prompt gets rated 4 or 5, and this time the switch routes to Gemini instead."

### Scene 4: The results (0:55 - 1:20)
*   **Visual:** `results/comparison_chart.png` and `results/comparison_table.md`, generated from a real 150-prompt run (75 GSM8K + 75 MMLU).
*   **Voiceover:** "Across 150 real prompts, routing cut average latency by 49% and slightly improved accuracy — 89.3% vs. 88.7% — by sending only 6 of 150 prompts to the expensive tier."
*   **Action:** Zoom on the Average Latency row and the Capable/Cheap Model Call counts.
*   **Voiceover:** "It didn't win on raw dollar cost this run — the cheap model answers more verbosely per token, which is itself an honest, real finding worth showing, not hiding. All code and logs are on GitHub."
