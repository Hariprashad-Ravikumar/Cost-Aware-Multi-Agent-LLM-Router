# RouteDemo: Video Demo Storyboard

**Target Length:** 60 - 90 seconds.
**Goal:** Show a clear, visual comparison between a dumb baseline approach and our smart, cost-aware routing pipeline.

### Scene 1: Introduction (0:00 - 0:15)
*   **Visual:** Show the `README.md` Architecture Mermaid Diagram.
*   **Voiceover:** "When building LLM apps, using the most capable model for every prompt is expensive. Here's a demo of a cost-aware routing pipeline that fixes this, using n8n, Groq, and Gemini."
*   **Action:** Briefly highlight the two paths in the diagram (Cheap vs Capable).

### Scene 2: n8n Workflow (0:15 - 0:35)
*   **Visual:** Switch to the n8n visual editor showing `route_demo.json`.
*   **Voiceover:** "We ingest prompts via webhook. A fast, cheap Groq model acts as our classifier, rating difficulty from 1 to 5. A switch node routes easy questions to a larger Groq model, and hard questions—like complex math—to Gemini 1.5 Pro."
*   **Action:** Click open the Switch node to show the `Score <= 3` and `Score >= 4` rules.

### Scene 3: Running the Pipeline (0:35 - 0:50)
*   **Visual:** Terminal window (split screen if possible).
*   **Action:** Run `python scripts/run_baseline.py` on the top half (or fast forward). Run `python scripts/run_routed.py` on the bottom half.
*   **Voiceover:** "We run a 150-prompt dataset mixing MMLU and GSM8K against both a standard baseline and our n8n routed pipeline. Everything is logged to CSVs."

### Scene 4: The Results (0:50 - 1:15)
*   **Visual:** Open `results/comparison_chart.png` and `results/comparison_table.md`.
*   **Voiceover:** "The results are clear. The routed pipeline maintains nearly identical accuracy, but slashes the total cost significantly by offloading the easier MMLU tasks to Groq."
*   **Action:** Zoom in on the Total Cost ($) difference and the Capable/Cheap Model Call counts.
*   **Voiceover:** "All code is available on GitHub to run locally."
