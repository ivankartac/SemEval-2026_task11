import asyncio
import json
import logging
import re

import ollama
from jinja2 import Template

from src.utils import split_syllogism

logger = logging.getLogger(__name__)


class EndToEndPredictor:
    def __init__(
        self,
        model: str,
        template: Template,
        api_url: str = None,
        timeout_seconds: int = 300,
        retry_attempts: int = 3,
        num_ctx: int = 16384,
        num_predict: int = 16384,
    ):
        self.model = model
        self.template = template
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.num_ctx = num_ctx
        self.num_predict = num_predict

        if api_url:
            self.client = ollama.AsyncClient(host=api_url)
        else:
            self.client = ollama.AsyncClient()

    async def __call__(self, example: dict) -> dict | None:
        syllogism = example["syllogism"]
        prompt = self.template.render(syllogism=syllogism)

        logger.info(f"Example {example['id']} (valid: {example.get('validity', 'N/A')})")

        content = None
        for attempt in range(self.retry_attempts):
            temperature = 0.6 if attempt > 0 else 0.0
            try:
                response = await asyncio.wait_for(
                    self.client.chat(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        options={
                            "temperature": temperature,
                            "num_ctx": self.num_ctx,
                            "num_predict": self.num_predict,
                        },
                    ),
                    timeout=self.timeout_seconds,
                )
                content = response.message.content
                break
            except asyncio.TimeoutError:
                logger.warning(f"  Attempt {attempt + 1} timed out.")
            except Exception as e:
                logger.warning(f"  Attempt {attempt + 1} failed: {e}")

        if content is None:
            logger.error(f"  All attempts failed for {example['id']}")
            return {
                "id": example["id"],
                "valid": None,
                "syllogism": split_syllogism(syllogism),
                "llm_response": None,
            }

        # Parse JSON from response
        valid = None
        reasoning = ""
        relevant_premises = []
        try:
            # Try to extract JSON from the response
            clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                parsed = json.loads(clean)
            valid = parsed.get("valid")
            if isinstance(valid, str):
                valid = valid.strip().lower() == "true"
            reasoning = parsed.get("reasoning", "")
            relevant_premises = parsed.get("relevant_premises", [])
        except (json.JSONDecodeError, AttributeError):
            # Fallback: look for true/false in the response
            lower = content.lower()
            if "true" in lower:
                valid = True
            elif "false" in lower:
                valid = False
            reasoning = content

        return {
            "id": example["id"],
            "valid": valid,
            "reasoning": reasoning,
            "syllogism": split_syllogism(syllogism),
            "llm_response": content,
            "relevant_premises": relevant_premises,
        }
