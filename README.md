# agentic-day3-production

A production-hardened customer support agent built with LangChain, demonstrating prompt engineering best practices from Week 2 Session 3.

## Features

- **Prompt templates as code** — system prompt lives in `prompts/support_agent_v1.yaml` with versioning and changelog
- **3-layer prompt injection defense** — input sanitization, hardened system prompt, output validation
- **Error handling with retries** — exponential backoff for rate limits, timeouts, and 5xx errors; immediate fail-fast for context overflow
- **Circuit breaker** — stops cascading failures after 5 consecutive errors; auto-recovers after 60 seconds
- **Structured JSON logging** — every event (token usage, retries, injection attempts, circuit state) emits a structured log line
- **Cost tracking** — per-call and session-total token counts and USD cost

## Setup

1. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**

   Create a `.env` file in the project root (**do not commit this file**):

   ```
   OPENAI_API_KEY=sk-...
   ```

4. **Run the agent**

   ```bash
   python app.py
   ```

## Project Structure

```
agentic-day3-production/
├── app.py                        # Main agent with all production hardening
├── prompts/
│   └── support_agent_v1.yaml     # Versioned prompt template
├── requirements.txt
├── .env                          # NOT committed — contains API keys
└── .gitignore
```

## Security Note

`.env` is listed in `.gitignore` and **must never be committed** to version control.
Add your `OPENAI_API_KEY` to `.env` locally before running.
