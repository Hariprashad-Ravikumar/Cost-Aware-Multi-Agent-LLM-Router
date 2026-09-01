"""Concurrency load test against the live /route endpoint.

Run: locust -f loadtest/locustfile.py --host http://localhost:8000 --headless \
       -u 10 -r 2 -t 60s --html results/load_test_report.html --csv results/load_test

Prompts are drawn from the real held-out eval set (data/eval_holdout.jsonl) - not
synthetic filler - so latency/cost numbers under load reflect the actual workload the
router was evaluated on, not an artificially easy or hard traffic pattern.
"""
import json
import random
from pathlib import Path

from locust import HttpUser, task, between

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_holdout.jsonl"
with open(DATA_PATH) as f:
    PROMPTS = [json.loads(line)["prompt"] for line in f]


class RouterUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def route_prompt(self):
        prompt = random.choice(PROMPTS)
        self.client.post("/route", json={"prompt": prompt}, name="/route", timeout=30)
