"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

log = logging.getLogger(__name__)

# gpt-4o-mini pricing (USD per token, as of 2024)
_COST_INPUT_PER_TOKEN = 0.15 / 1_000_000
_COST_OUTPUT_PER_TOKEN = 0.60 / 1_000_000


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client — backed by OpenAI chat completions."""

    def __init__(self) -> None:
        from openai import OpenAI  # lazy import keeps startup fast when not used

        settings = get_settings()

        # Priority: GROQ_API_KEY → OPENAI_API_KEY (gsk_ prefix) → OPENAI_API_KEY
        if settings.groq_api_key:
            api_key = settings.groq_api_key
            base_url = settings.openai_base_url or "https://api.groq.com/openai/v1"
            model = settings.groq_model
            log.info("Using Groq provider (GROQ_API_KEY).")
        elif settings.openai_api_key and settings.openai_api_key.startswith("gsk_"):
            api_key = settings.openai_api_key
            base_url = settings.openai_base_url or "https://api.groq.com/openai/v1"
            model = settings.openai_model
            log.info("Auto-detected Groq provider from OPENAI_API_KEY prefix.")
        elif settings.openai_api_key:
            api_key = settings.openai_api_key
            base_url = settings.openai_base_url  # None → default OpenAI
            model = settings.openai_model
        else:
            raise AgentExecutionError("No API key found. Set GROQ_API_KEY or OPENAI_API_KEY.")

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        log.info("LLMClient ready model=%s base_url=%s", self._model, base_url or "openai")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> LLMResponse:
        """Return a model completion with retry, timeout, and token tracking."""

        log.debug("LLM request model=%s sys_len=%d user_len=%d", self._model, len(system_prompt), len(user_prompt))
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                timeout=60,
            )
        except Exception as exc:
            log.error("LLM call failed: %s", exc)
            raise AgentExecutionError(f"LLM call failed: {exc}") from exc

        choice = response.choices[0]
        usage = response.usage
        in_tok = usage.prompt_tokens if usage else None
        out_tok = usage.completion_tokens if usage else None
        cost = None
        if in_tok is not None and out_tok is not None:
            cost = in_tok * _COST_INPUT_PER_TOKEN + out_tok * _COST_OUTPUT_PER_TOKEN

        log.debug("LLM response tokens in=%s out=%s cost_usd=%s", in_tok, out_tok, cost)
        return LLMResponse(
            content=choice.message.content or "",
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )
