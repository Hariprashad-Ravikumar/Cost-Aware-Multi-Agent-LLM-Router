# Release Notes: Calibrated Cost-Aware LLM Router v2.0

**Release Date:** September 1, 2026  
**Status:** Production Ready  
**Version:** v2.0

---

## 🎯 Overview

The Calibrated Cost-Aware LLM Router is a **production-grade FastAPI service** that intelligently routes LLM prompts across three model tiers using a calibrated logistic-regression router. This release achieves **93.9% cost reduction** compared to always using the most capable model, with **no measurable quality loss**.

---

## 📊 Key Results

| Metric | Routed (ε=0.15) | Always-Capable | Savings |
|--------|-----------------|-----------------|---------|
| **Cost (50 prompts)** | $0.00569 | $0.09387 | **93.9% reduction** |
| **Accuracy** | 96.0% (86.5-98.9% CI) | 92.0% (83.8-97.9% CI) | No measurable difference |
| **At 1M req/month** | $113.80/mo | $1,877.40/mo | **$21,163/year saved** |

---

## 🏗️ Architecture

### Three-Tier Model Stack

```
Prompt → Draft Call (Cheap Tier)
       ↓
Feature Extraction (4 signals)
       ↓
Calibrator (Logistic Regression)
       ↓
Error Budget Decision
├─→ P(correct) ≥ (1-ε) → Use Cheap Tier ($0.20/$1.25 per 1M)
├─→ Escalate → Mid Tier ($0.25/$1.50 per 1M)
└─→ Escalate Further → Capable Tier ($2.00/$10.00 per 1M)
```

### Models

| Tier | Model | Role | Pricing |
|------|-------|------|---------|
| **Cheap** | OpenAI `gpt-5.4-nano` | Primary answerer; feeds calibrator | $0.20/$1.25 per 1M |
| **Mid** | Google `gemini-3.1-flash-lite` | First escalation target | $0.25/$1.50 per 1M |
| **Capable** | Anthropic `claude-sonnet-5` | Top-of-ladder fallback | $2.00/$10.00 per 1M |

---

## 🔧 Technical Stack

### Core Service
- **Framework:** FastAPI
- **ORM:** SQLAlchemy + Alembic
- **Caching:** Redis (Upstash)
- **Database:** PostgreSQL (Neon)
- **Deployment:** Google Cloud Run
- **Monitoring:** Prometheus
- **Resilience:** Hand-rolled async circuit breaker

### Calibration & ML
- **Calibrator:** scikit-learn logistic regression
- **4 Features:**
  - Logprob uncertainty
  - Self-consistency dispersion
  - Hard-cluster embedding distance
  - Response length
- **Validation:** Expected Calibration Error (ECE), Brier score
- **Benchmarks:** GSM8K + MMLU

### Quality Assurance
- **Tests:** 16 pytest tests covering decision logic, stats utilities, circuit breaker
- **Live Demo:** https://calibrated-router-768949786238.us-central1.run.app

---

## 🚀 Real Engineering Findings

This release documents actual bugs and gotchas encountered in production:

1. **Broken Third-Party Circuit Breaker** — `pybreaker.CircuitBreaker.call_async` had async safety issues; replaced with hand-rolled implementation
2. **Rate Limit Discovery** — Bulk calibration data collection triggered 429 cascades; implemented exponential backoff
3. **Feature Engineering Bug** — Self-consistency comparison failed on multi-step explanations; fixed with normalized comparison
4. **Tier Pricing Inversion** — Mid-tier model was accidentally more expensive; swapped Claude Haiku ↔ Gemini Flash-Lite
5. **Cloud Run Memory Leak** — Initial deploy exceeded 1024 MiB on held-out eval; refactored to streaming inference

---

## 📁 Files & Directories

```
.
├── README.md                      # Main documentation
├── VERSION                        # Version file (V2)
├── .github/repo-settings.json     # Repository metadata
├── v2/                            # Production service (FastAPI)
│   ├── main.py                    # FastAPI app entry point
│   ├── requirements.txt           # Python dependencies
│   ├── CASE_STUDY.md              # Detailed engineering writeup
│   ├── app/                       # Service modules
│   │   ├── main.py
│   │   ├── models.py
│   │   ├── calibrator.py
│   │   ├── router.py
│   │   └── circuit_breaker.py
│   ├── scripts/
│   │   ├── build_eval_sets.py
│   │   ├── collect_calibration_data.py
│   │   ├── train_calibrator.py
│   │   ├── run_eval.py
│   │   └── budget_sweep.py
│   ├── results/
│   │   ├── eval_report.md
│   │   └── budget_sweep.md
│   └── loadtest/
│       └── locustfile.py
├── v1/                            # Earlier prototype (n8n-based)
│   └── README.md
└── RELEASE_NOTES_V2.md            # This file
```

---

## 🔄 Upgrade from v1

**v1** (n8n-orchestrated, prompted classifier) is superseded by **v2** (FastAPI, calibrated router).

- **v1 Findings:** Routing won on latency and accuracy, but cost was neutral on this benchmark
- **v2 Improvements:** Calibrated router achieves 93.9% cost savings with better statistical rigor

The v1 code remains in the repository root for reference; v2 is the active production system.

---

## ⚙️ Setup & Running

### 1. Environment Setup
```bash
cd v2
cp .env.example .env
# Add OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Infrastructure
```bash
# Point DATABASE_URL / REDIS_URL in .env to your Postgres/Redis
alembic upgrade head
```

### 3. Build & Train
```bash
python scripts/build_eval_sets.py
python scripts/collect_calibration_data.py
python scripts/train_calibrator.py
```

### 4. Run Service
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# Visit http://localhost:8000/ for interactive demo
# POST /route for direct API calls
```

### 5. Evaluate & Benchmark
```bash
python scripts/run_eval.py
python scripts/budget_sweep.py
locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 8 -r 2 -t 45s
```

**Test Mode:** Set `TEST_MODE=true` to run on 5 prompts first.

---

## 📈 Evaluation Methodology

- **Held-out Set:** 50 prompts from `data/eval_holdout.jsonl` (never used in calibrator training)
- **Benchmarks:** GSM8K (math word problems) + MMLU (multitask reasoning)
- **Metrics:** Cost, accuracy with 95% confidence intervals, ECE, Brier score
- **Error Budgets:** Sweep from ε=0.05 to ε=0.50 in `results/budget_sweep.md`
- **Ablation:** Comparison against naive prompted classifier in `results/eval_report.md`

---

## 🔒 Honest Limitations

1. **Train/Serve Feature Skew** — Self-consistency sampling (3 extra calls) is offline-only; serve-time router uses only draft call
2. **Label Quality** — Correctness labels from established benchmarks (GSM8K, MMLU), not error-free
3. **Model Drift** — Calibrator is static; no automatic retraining on model/usage drift
4. **Sample Size** — n=50 in held-out eval; confidence intervals are wide (86.5%-98.9% for ε=0.15)
5. **Cheap-Tier Volatility** — OpenAI `gpt-5.4-nano` changed during development; validate on your deployment

---

## 📚 Prior Art & References

Not a claim of novel research — this is a careful engineering reproduction of established patterns:

1. **FrugalGPT** (Chen et al., 2023): Cost-aware routing framework
2. **RouteLLM** (LMSYS): Learned routing over prompted classification
3. **RouterBench** (2024): Routing benchmarking patterns
4. **Calibration Literature** (Guo et al., 2017): Expected Calibration Error
5. **Self-Consistency** (Wang et al., 2022): Multi-sample reasoning validation

See `v2/CASE_STUDY.md` for detailed citations and divergence from FrugalGPT's full method.

---

## 🐛 Known Issues & Workarounds

| Issue | Workaround |
|-------|-----------|
| `logprobs=True` unsupported on Groq | Use OpenAI tier instead; validated before serving |
| Rate limits (8000 tok/min on cheap tier) | Implement exponential backoff; see circuit breaker |
| Cloud Run memory: 1024 MiB insufficient | Increased to 2048 MiB; streaming inference for large batches |
| Model pricing inversions | Always validate pricing table before deployment |

---

## 🚢 Deployment

### Cloud Run (Production)

```bash
cd v2
gcloud run deploy calibrated-router \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 2048Mi \
  --set-env-vars OPENAI_API_KEY=$OPENAI_API_KEY,GEMINI_API_KEY=$GEMINI_API_KEY,ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY,DATABASE_URL=$DATABASE_URL,REDIS_URL=$REDIS_URL
```

**Live Instance:** https://calibrated-router-768949786238.us-central1.run.app  
(Cloud Run + Neon Postgres + Upstash Redis, backed by real production data)

---

## 📞 Support & Contribution

- **Issues & Feedback:** GitHub Issues
- **Questions on Setup:** See `v2/README.md` and `.env.example`
- **Benchmarking:** Reproduce with `python scripts/run_eval.py`
- **Customization:** Error budget `ε` is tunable in `app/config.py`

---

## 📝 License

See repository root for license details.

---

## ✅ Checklist for v2.0 Production Release

- [x] Calibrated router achieves 93.9% cost reduction
- [x] Accuracy validated on held-out eval (96.0% vs. 92.0% baseline, no statistically significant difference)
- [x] Circuit breaker hand-rolled and tested
- [x] Cloud Run deployment live and stable
- [x] Postgres + Redis caching operational
- [x] 16 tests passing (decision logic, stats, circuit breaker)
- [x] Prometheus monitoring integrated
- [x] Real engineering findings documented
- [x] CASE_STUDY.md with honest limitations
- [x] Version file (V2) added
- [x] Repository description & topics updated

---

**Released by:** Hariprashad Ravikumar  
**Repository:** https://github.com/Hariprashad-Ravikumar/Cost-Aware-Multi-Agent-LLM-Router  
**Questions?** See `CASE_STUDY.md` for deep technical details.
