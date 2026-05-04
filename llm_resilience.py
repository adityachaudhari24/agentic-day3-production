import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List

logger = logging.getLogger("support_agent")


class ErrorCategory(str, Enum):
    RATE_LIMIT = "RATE_LIMIT"
    TIMEOUT = "TIMEOUT"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    AUTH_ERROR = "AUTH_ERROR"
    UNKNOWN = "UNKNOWN"


@dataclass
class InvocationResult:
    success: bool
    content: str = ""
    error: str = ""
    error_category: ErrorCategory = ErrorCategory.UNKNOWN
    attempts: int = 0


def production_invoke(messages: list, llm, max_retries: int = 3) -> InvocationResult:
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.invoke(messages)
            return InvocationResult(
                success=True,
                content=response.content,
                attempts=attempt,
            )
        except Exception as e:
            message = str(e).lower()

            if "rate limit" in message or "429" in message:
                if attempt < max_retries:
                    delay = 2 ** attempt  # 2s, 4s, 8s
                    logger.warning(f"Rate limited. Retry {attempt}/{max_retries} in {delay}s")
                    time.sleep(delay)
                    continue
                return InvocationResult(
                    success=False,
                    error="Service temporarily unavailable. Please try again later.",
                    error_category=ErrorCategory.RATE_LIMIT,
                    attempts=attempt,
                )

            if "timeout" in message or "timed out" in message:
                if attempt < max_retries:
                    logger.warning(f"Timeout. Retry {attempt}/{max_retries}")
                    time.sleep(1)
                    continue
                return InvocationResult(
                    success=False,
                    error="Request timed out. Please try again.",
                    error_category=ErrorCategory.TIMEOUT,
                    attempts=attempt,
                )

            if "context_length" in message or "maximum context length" in message or "tokens" in message:
                logger.error("Context overflow — trim conversation history (no retry)")
                return InvocationResult(
                    success=False,
                    error="Conversation too long. Please start a new session.",
                    error_category=ErrorCategory.CONTEXT_OVERFLOW,
                    attempts=attempt,
                )

            if "invalid_api_key" in message or "401" in message:
                logger.critical("Auth error — check API keys (no retry)")
                return InvocationResult(
                    success=False,
                    error="Service temporarily unavailable. Please try again later.",
                    error_category=ErrorCategory.AUTH_ERROR,
                    attempts=attempt,
                )

            logger.error(f"Unknown error (attempt {attempt}): {type(e).__name__}: {e}")
            if attempt < max_retries:
                time.sleep(0.5)
                continue
            return InvocationResult(
                success=False,
                error="Service temporarily unavailable. Our team has been notified.",
                error_category=ErrorCategory.UNKNOWN,
                attempts=attempt,
            )

    return InvocationResult(
        success=False,
        error="Max retries exceeded.",
        error_category=ErrorCategory.UNKNOWN,
        attempts=max_retries,
    )


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    reset_timeout: float = 60.0
    failures: int = 0
    state: str = "closed"  # "closed" | "open" | "half-open"
    last_failure_time: float = field(default_factory=time.time)

    def allow_request(self) -> bool:
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
                logger.info("[CircuitBreaker] Half-open — allowing one test request")
                return True
            logger.warning("[CircuitBreaker] OPEN — request blocked")
            return False
        return True

    def record_success(self) -> None:
        self.failures = 0
        self.state = "closed"
        logger.info("[CircuitBreaker] Success — state CLOSED")

    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"[CircuitBreaker] OPEN — {self.failures} failures, "
                f"blocking for {self.reset_timeout}s"
            )


# Singleton breaker shared across all requests in this process
_breaker = CircuitBreaker()


def guarded_invoke(messages: list, llm, max_retries: int = 3) -> InvocationResult:
    if not _breaker.allow_request():
        return InvocationResult(
            success=False,
            error="Service temporarily unavailable. Please try again in a few minutes.",
            error_category=ErrorCategory.UNKNOWN,
            attempts=0,
        )

    result = production_invoke(messages, llm, max_retries=max_retries)

    if result.success:
        _breaker.record_success()
    else:
        _breaker.record_failure()

    return result
