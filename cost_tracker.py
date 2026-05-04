import json
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PRICING = {
    "gpt-4o-mini": {"input": 0.000015, "output": 0.00006},  # per 1K tokens
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = PRICING.get(model, PRICING["gpt-4o-mini"])
    return (input_tokens * prices["input"] / 1000) + (
        output_tokens * prices["output"] / 1000
    )


@dataclass
class SessionCostTracker:
    session_id: str
    model: str = "gpt-4o-mini"
    budget_usd: float = 0.50
    total_cost_usd: float = 0.0
    call_count: int = 0

    def log_call(
        self, input_tokens: int, output_tokens: int, latency_ms: float, success: bool
    ) -> None:
        cost = calculate_cost(self.model, input_tokens, output_tokens)
        self.total_cost_usd += cost
        self.call_count += 1
        logger.info(
            json.dumps(
                {
                    "event": "llm_call",
                    "session_id": self.session_id,
                    "model": self.model,
                    "cost_usd": round(cost, 6),
                    "session_total_usd": round(self.total_cost_usd, 6),
                    "latency_ms": latency_ms,
                    "success": success,
                }
            )
        )

    def check_budget(self) -> bool:
        """Return True if under budget, False if exceeded."""
        return self.total_cost_usd < self.budget_usd


def budget_aware_invoke(tracker: SessionCostTracker, messages: list) -> str:
    """Invoke the LLM with budget enforcement and cost tracking."""
    from llm_resilience import guarded_invoke
    from langchain_openai import ChatOpenAI

    if not tracker.check_budget():
        return "I've reached my session limit. Please start a new session."

    llm = ChatOpenAI(model=tracker.model, temperature=0)
    start = time.time()
    result = guarded_invoke(messages, llm)
    latency = round((time.time() - start) * 1000, 2)

    tracker.log_call(
        input_tokens=100,
        output_tokens=50,
        latency_ms=latency,
        success=result.success,
    )

    return result.content if result.success else "Something went wrong."
