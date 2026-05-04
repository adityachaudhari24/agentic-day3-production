"""
Runs each production-error scenario via a MOCK invoker so you can see behavior
without triggering real API rate limits or waiting for timeouts.

Scenarios:
  1. Normal success
  2. Rate limit → exponential backoff → retry → success
  3. Timeout → retry → success (or fail after max retries)
  4. Context length overflow → no retry, return trim message
  5. Auth error (401) → no retry, user-friendly message
  6. Unknown error → retry until max_retries then fail
  7. Circuit breaker: N failures → OPEN → block requests → reset → half-open → success
  8. Circuit breaker state NOT persisted between requests (singleton/external store in prod)

Run: python d15_production_error_scenarios.py
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

from langchain_core.messages import BaseMessage

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("production_agent")


# ===== ERROR CATEGORIES =====
class ErrorCategory(Enum):
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONTEXT_OVERFLOW = "context_overflow"
    INVALID_REQUEST = "invalid_request"
    AUTH_ERROR = "auth_error"
    UNKNOWN = "unknown"


@dataclass
class InvocationResult:
    success: bool
    content: str = ""
    error: str = ""
    error_category: ErrorCategory = ErrorCategory.UNKNOWN
    attempts: int = 0
    latency_ms: float = 0.0


# ===== MOCK INJECTABLE INVOKER =====
def production_invoke(
    messages: List[BaseMessage],
    max_retries: int = 3,
    invoke_fn: Optional[Callable[[List[BaseMessage]], str]] = None,
    retry_delay_scale: float = 1.0,
) -> InvocationResult:
    """
    Production-grade LLM invocation with pluggable invoker.
    When invoke_fn is provided (e.g. mock), use it instead of real LLM.
    """
    for attempt in range(1, max_retries + 1):
        start = time.time()
        try:
            if invoke_fn is not None:
                content = invoke_fn(messages)
            else:
                # Real path: would use ChatOpenAI here
                from langchain_openai import ChatOpenAI
                from dotenv import load_dotenv
                load_dotenv()
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, request_timeout=30)
                response = llm.invoke(messages)
                content = response.content

            latency = round((time.time() - start) * 1000, 2)
            return InvocationResult(
                success=True,
                content=content,
                attempts=attempt,
                latency_ms=latency,
            )

        except Exception as e:
            latency = round((time.time() - start) * 1000, 2)
            error_str = str(e).lower()

            if "rate limit" in error_str or "429" in error_str:
                category = ErrorCategory.RATE_LIMIT
                if attempt < max_retries:
                    delay = (2 ** attempt) * retry_delay_scale
                    logger.warning(
                        f"  Rate limited. Retry {attempt}/{max_retries} in {delay}s"
                    )
                    time.sleep(delay)
                    continue

            elif "timeout" in error_str or "timed out" in error_str:
                category = ErrorCategory.TIMEOUT
                if attempt < max_retries:
                    logger.warning(f"  Timeout. Retry {attempt}/{max_retries}")
                    time.sleep(0.3)  # short delay in demo
                    continue

            elif "context_length" in error_str or "tokens" in error_str:
                category = ErrorCategory.CONTEXT_OVERFLOW
                logger.error("  Context overflow — trim conversation history (no retry)")
                return InvocationResult(
                    success=False,
                    error="Conversation too long. Starting a new session.",
                    error_category=category,
                    attempts=attempt,
                    latency_ms=latency,
                )

            elif "invalid_api_key" in error_str or "401" in error_str:
                category = ErrorCategory.AUTH_ERROR
                logger.critical("  AUTH ERROR — check API keys (no retry)")
                return InvocationResult(
                    success=False,
                    error="Service temporarily unavailable. Please try again later.",
                    error_category=category,
                    attempts=attempt,
                    latency_ms=latency,
                )

            else:
                category = ErrorCategory.UNKNOWN
                logger.error(
                    f"  Unknown error (attempt {attempt}): {type(e).__name__}: {e}"
                )
                if attempt < max_retries:
                    time.sleep(0.2)
                    continue

            return InvocationResult(
                success=False,
                error="Service temporarily unavailable. Our team has been notified.",
                error_category=category,
                attempts=attempt,
                latency_ms=latency,
            )

    return InvocationResult(
        success=False,
        error="Max retries exceeded.",
        attempts=max_retries,
    )


# ===== CIRCUIT BREAKER =====
@dataclass
class CircuitBreaker:
    """
    Stops calling the LLM after too many consecutive failures (open state).

    IMPORTANT — State must be shared across requests:
    - State is NOT persisted between requests by default (in-memory only).
    - In production: use a SINGLETON (one instance per process) or store state
      externally (e.g. Redis) so all requests see the same failure count.
    - If you create a new CircuitBreaker per request (e.g. in a stateless API),
      the circuit never opens because each request starts with failures=0.
    """
    failure_threshold: int = 3
    reset_timeout: float = 2.0
    failures: int = 0
    last_failure_time: float = 0.0
    state: str = "closed"

    def allow_request(self) -> bool:
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
                logger.info("  [CircuitBreaker] Half-open — allowing one test request")
                return True
            logger.info("  [CircuitBreaker] OPEN — request blocked")
            return False
        return True

    def record_success(self):
        self.failures = 0
        self.state = "closed"
        logger.info("  [CircuitBreaker] Success — state CLOSED")

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"  [CircuitBreaker] OPEN — {self.failures} failures, "
                f"blocking for {self.reset_timeout}s"
            )


def circuit_protected_invoke(
    messages: List[BaseMessage],
    breaker: CircuitBreaker,
    invoke_fn: Optional[Callable[[List[BaseMessage]], str]] = None,
    retry_delay_scale: float = 1.0,
) -> str:
    if not breaker.allow_request():
        return "Service temporarily unavailable. Please try again in a few minutes."

    result = production_invoke(
        messages, max_retries=1, invoke_fn=invoke_fn, retry_delay_scale=retry_delay_scale
    )

    if result.success:
        breaker.record_success()
        return result.content
    else:
        breaker.record_failure()
        return result.error


# ===== MOCK INVOKER BUILDER =====
def make_mock_invoker(
    failures: List[Exception],
    success_response: str = "Mock LLM response.",
):
    """Returns a callable that raises each failure in order, then returns success_response."""
    call_count = [0]

    def invoker(messages: List[BaseMessage]) -> str:
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(failures):
            raise failures[idx]
        return success_response

    return invoker


# Short delay scale for demo (0.1 → 0.2s, 0.4s, 0.8s instead of 2s, 4s, 8s)
DEMO_DELAY_SCALE = 0.1


# ===== SCENARIO RUNNERS =====
def scenario_1_normal_success():
    print("\n" + "=" * 60)
    print("SCENARIO 1: Normal success")
    print("=" * 60)
    invoker = make_mock_invoker([], success_response="Paris is the capital of France.")
    result = production_invoke([], invoke_fn=invoker, retry_delay_scale=DEMO_DELAY_SCALE)
    print(f"  Success: {result.success}, Attempts: {result.attempts}, Content: {result.content[:50]}...")
    assert result.success and result.attempts == 1


def scenario_2_rate_limit_then_success():
    print("\n" + "=" * 60)
    print("SCENARIO 2: Rate limit (429) → backoff → retry → success")
    print("=" * 60)
    invoker = make_mock_invoker(
        [Exception("Rate limit exceeded. 429")],
        success_response="Here is your answer after retry.",
    )
    result = production_invoke(
        [], max_retries=3, invoke_fn=invoker, retry_delay_scale=DEMO_DELAY_SCALE
    )
    print(f"  Success: {result.success}, Attempts: {result.attempts}")
    print(f"  Content: {result.content}")
    assert result.success and result.attempts == 2


def scenario_3_timeout_then_success():
    print("\n" + "=" * 60)
    print("SCENARIO 3: Timeout → retry → success")
    print("=" * 60)
    invoker = make_mock_invoker(
        [Exception("Request timed out after 30s")],
        success_response="Response after timeout retry.",
    )
    result = production_invoke(
        [], max_retries=3, invoke_fn=invoker, retry_delay_scale=DEMO_DELAY_SCALE
    )
    print(f"  Success: {result.success}, Attempts: {result.attempts}")
    assert result.success and result.attempts == 2


def scenario_4_context_overflow_no_retry():
    print("\n" + "=" * 60)
    print("SCENARIO 4: Context length overflow → no retry, return trim message")
    print("=" * 60)
    invoker = make_mock_invoker(
        [Exception("Context length exceeded. Maximum tokens.")],
        success_response="",
    )
    result = production_invoke(
        [], max_retries=3, invoke_fn=invoker, retry_delay_scale=DEMO_DELAY_SCALE
    )
    print(f"  Success: {result.success}, Category: {result.error_category.value}")
    print(f"  Error (user message): {result.error}")
    assert not result.success and result.error_category == ErrorCategory.CONTEXT_OVERFLOW
    assert result.attempts == 1  # no retry


def scenario_5_auth_error_no_retry():
    print("\n" + "=" * 60)
    print("SCENARIO 5: Auth error (401) → no retry, user-friendly message")
    print("=" * 60)
    invoker = make_mock_invoker(
        [Exception("Invalid API key. 401 Unauthorized")],
        success_response="",
    )
    result = production_invoke(
        [], max_retries=3, invoke_fn=invoker, retry_delay_scale=DEMO_DELAY_SCALE
    )
    print(f"  Success: {result.success}, Category: {result.error_category.value}")
    print(f"  Error: {result.error}")
    assert not result.success and result.error_category == ErrorCategory.AUTH_ERROR
    assert result.attempts == 1


def scenario_6_unknown_error_retry_then_fail():
    print("\n" + "=" * 60)
    print("SCENARIO 6: Unknown error → retry until max_retries then fail")
    print("=" * 60)
    invoker = make_mock_invoker(
        [
            Exception("Connection reset by peer"),
            Exception("Connection reset by peer"),
            Exception("Connection reset by peer"),
        ],
        success_response="",
    )
    result = production_invoke(
        [], max_retries=3, invoke_fn=invoker, retry_delay_scale=DEMO_DELAY_SCALE
    )
    print(f"  Success: {result.success}, Attempts: {result.attempts}")
    print(f"  Final error: {result.error}")
    assert not result.success and result.attempts == 3


def scenario_7_circuit_breaker():
    print("\n" + "=" * 60)
    print("SCENARIO 7: Circuit breaker — failures → OPEN → block → half-open → success")
    print("=" * 60)
    print("  (Using ONE shared CircuitBreaker instance — state persists across calls.)")
    breaker = CircuitBreaker(failure_threshold=3, reset_timeout=2.0)

    # Invoker: fail 3 times (one per circuit_protected_invoke call, max_retries=1), then succeed
    invoker = make_mock_invoker(
        [
            Exception("Rate limit 429"),
            Exception("Rate limit 429"),
            Exception("Rate limit 429"),
        ],
        success_response="Recovered response.",
    )

    print("  Step 1: Three failing calls to open the circuit...")
    for i in range(3):
        out = circuit_protected_invoke(
            [], breaker, invoke_fn=invoker, retry_delay_scale=DEMO_DELAY_SCALE
        )
        print(f"    Call {i+1}: {out[:60]}...")

    print("  Step 2: Next call should be BLOCKED (circuit open)...")
    out_blocked = circuit_protected_invoke(
        [], breaker, invoke_fn=invoker, retry_delay_scale=DEMO_DELAY_SCALE
    )
    print(f"    Blocked response: {out_blocked}")

    print("  Step 3: Wait 2s for reset_timeout, then one call should be allowed (half-open)...")
    time.sleep(2.2)
    out_recovered = circuit_protected_invoke(
        [], breaker, invoke_fn=invoker, retry_delay_scale=DEMO_DELAY_SCALE
    )
    print(f"    After reset: {out_recovered}")

    print("  Step 4: Next call should succeed (circuit closed again)...")
    out_ok = circuit_protected_invoke(
        [], breaker, invoke_fn=invoker, retry_delay_scale=DEMO_DELAY_SCALE
    )
    print(f"    Next: {out_ok}")
    assert "Recovered" in out_recovered or "Recovered" in out_ok


def scenario_8_rate_limit_exhaust_retries():
    print("\n" + "=" * 60)
    print("SCENARIO 8: Rate limit every time → exhaust retries → fail")
    print("=" * 60)
    invoker = make_mock_invoker(
        [
            Exception("429 Rate limit"),
            Exception("429 Rate limit"),
            Exception("429 Rate limit"),
        ],
        success_response="",
    )
    result = production_invoke(
        [], max_retries=3, invoke_fn=invoker, retry_delay_scale=DEMO_DELAY_SCALE
    )
    print(f"  Success: {result.success}, Attempts: {result.attempts}")
    print(f"  Category: {result.error_category.value}, Error: {result.error[:50]}...")
    assert not result.success and result.attempts == 3
    assert result.error_category == ErrorCategory.RATE_LIMIT


def scenario_9_circuit_breaker_state_must_be_shared():
    """
    Demonstrates: circuit breaker state NOT persisted per request.
    If you create a new CircuitBreaker per request (e.g. stateless API),
    the circuit never opens — each request sees failures=0.
    """
    print("\n" + "=" * 60)
    print("SCENARIO 9: Circuit breaker state must be SHARED (singleton/external)")
    print("=" * 60)

    invoker = make_mock_invoker(
        [
            Exception("429"),
            Exception("429"),
            Exception("429"),
        ],
        success_response="OK",
    )

    print("  WRONG: New CircuitBreaker per 'request' (e.g. stateless server)...")
    for i in range(3):
        breaker = CircuitBreaker(failure_threshold=3, reset_timeout=2.0)
        out = circuit_protected_invoke(
            [], breaker, invoke_fn=invoker, retry_delay_scale=DEMO_DELAY_SCALE
        )
        print(f"    Request {i+1}: new breaker, failures={breaker.failures}, state={breaker.state!r}")
    print("  → Circuit never opened: each request had its own fresh breaker (failures=0).")

    print()
    print("  RIGHT: ONE shared CircuitBreaker (singleton or Redis) across requests...")
    shared_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=2.0)
    invoker2 = make_mock_invoker(
        [Exception("429"), Exception("429"), Exception("429")],
        success_response="OK",
    )
    for i in range(4):
        out = circuit_protected_invoke(
            [], shared_breaker, invoke_fn=invoker2, retry_delay_scale=DEMO_DELAY_SCALE
        )
        # "Please try again in a few minutes" = circuit open, request never reached LLM
        blocked = "Please try again in a few minutes" in out
        print(f"    Request {i+1}: state={shared_breaker.state!r} → {'BLOCKED (no LLM call)' if blocked else 'invoked LLM'}")
    print("  → After 3 failures circuit OPEN; 4th request is blocked (no LLM call).")


# ===== MAIN =====
if __name__ == "__main__":
    print("\n  DEMO 15 — Production Error Handling (all scenarios)", flush=True)
    print("  Using MOCK invoker — no real API calls for rate limit/timeout.\n", flush=True)

    scenario_1_normal_success()
    scenario_2_rate_limit_then_success()
    scenario_3_timeout_then_success()
    scenario_4_context_overflow_no_retry()
    scenario_5_auth_error_no_retry()
    scenario_6_unknown_error_retry_then_fail()
    scenario_7_circuit_breaker()
    scenario_8_rate_limit_exhaust_retries()
    scenario_9_circuit_breaker_state_must_be_shared()

    print("\n" + "=" * 60)
    print("All scenarios completed.")
    print("=" * 60 + "\n")
