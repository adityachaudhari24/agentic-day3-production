"""
DEMO 16: Cost Tracking & Token Budgets
---------------------------------------
What you'll see:
  1. Tracking token usage per call
  2. Calculating cost in USD
  3. Session-level budget enforcement
  4. Structured logs for cost analysis

Key concepts: usage_metadata, cost tracking, budget controls
"""

import json
import logging
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("cost_tracker")

# ===== PRICING (as of early 2025 — update periodically) =====
PRICING = {
    "gpt-4o-mini": {"input": 0.000015, "output": 0.00006},  # per 1K tokens
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a single LLM call."""
    prices = PRICING.get(model, PRICING["gpt-4o-mini"])
    return (input_tokens * prices["input"] / 1000) + (
        output_tokens * prices["output"] / 1000
    )


# ===== SESSION TRACKER =====
@dataclass
class SessionCostTracker:
    session_id: str
    model: str = "gpt-4o-mini"
    budget_usd: float = 0.10  # $0.10 default budget per session
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    call_count: int = 0
    calls: list = field(default_factory=list)

    def log_call(
        self, input_tokens: int, output_tokens: int, latency_ms: float, success: bool
    ):
        cost = calculate_cost(self.model, input_tokens, output_tokens)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        self.call_count += 1

        call_log = {
            "call_number": self.call_count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
            "latency_ms": latency_ms,
            "success": success,
            "session_total_cost": round(self.total_cost_usd, 6),
        }
        self.calls.append(call_log)
        logger.info(
            json.dumps({"event": "llm_call", "session_id": self.session_id, **call_log})
        )

    def check_budget(self) -> bool:
        """Returns True if under budget, False if exceeded."""
        if self.total_cost_usd >= self.budget_usd:
            logger.warning(
                json.dumps(
                    {
                        "event": "budget_exceeded",
                        "session_id": self.session_id,
                        "spent_usd": round(self.total_cost_usd, 6),
                        "budget_usd": self.budget_usd,
                    }
                )
            )
            return False
        return True

    def summary(self):
        return {
            "session_id": self.session_id,
            "total_calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "budget_usd": self.budget_usd,
            "budget_remaining": round(self.budget_usd - self.total_cost_usd, 6),
            "budget_used_pct": round((self.total_cost_usd / self.budget_usd) * 100, 1)
            if self.budget_usd
            else 0.0,
        }


# ===== BUDGET-AWARE AGENT =====
def budget_aware_invoke(
    tracker: SessionCostTracker,
    messages: list,
    fallback: str = "I've reached my session limit. Please start a new session.",
) -> str:
    """Invokes LLM with cost tracking and budget enforcement."""
    if not tracker.check_budget():
        return fallback

    llm = ChatOpenAI(model=tracker.model, temperature=0) # initialize the model
    start = time.time() # start the timer

    try:
        response = llm.invoke(messages)
        latency = round((time.time() - start) * 1000, 2)
        usage = response.usage_metadata or {}
        tracker.log_call(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_ms=latency,
            success=True,
        )
        return response.content
    except Exception:
        latency = round((time.time() - start) * 1000, 2)
        tracker.log_call(0, 0, latency, success=False)
        raise


if __name__ == "__main__":
    # ===== SIMULATE A SESSION =====
    session = SessionCostTracker(
        session_id="user_123_session_456", budget_usd=0.01
    )

    questions = [
        "What is machine learning?",
        "Explain neural networks in one sentence.",
        "What is a transformer model?",
    ]

    for q in questions:
        response = budget_aware_invoke(
            tracker=session,
            messages=[
                SystemMessage(
                    content="You are a concise AI expert. Answer in 1-2 sentences."
                ),
                HumanMessage(content=q),
            ],
        )
        print(f"Q: {q}")
        print(f"A: {response}\n")

    # ===== PRINT SUMMARY =====
    print("=" * 50)
    print("SESSION COST SUMMARY:")
    print(json.dumps(session.summary(), indent=2))

