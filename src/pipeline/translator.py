import asyncio
import json
import logging

import ollama
from jinja2 import Template

from src.utils import split_syllogism

logger = logging.getLogger(__name__)


class Translator:
    def __init__(
        self,
        model: str,
        translate_template: Template,
        evaluate_template: Template,
        api_url: str = None,
        num_ctx: int = 4096,
        temperature: float = 0.0,
        timeout_seconds: int = 300,
    ):
        self.model = model
        self.translate_template = translate_template
        self.evaluate_template = evaluate_template
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

        if api_url:
            self.client = ollama.AsyncClient(host=api_url)
        else:
            self.client = ollama.AsyncClient()

    async def __call__(self, syllogism: str) -> str:
        """Translate a syllogism to English, with evaluation and retry.

        Args:
            syllogism: The syllogism text to translate.

        Returns:
            The translated syllogism text.
        """
        prompt = self._render_translate_prompt(syllogism)
        translation = await self._chat(prompt)
        translation = translation.replace("\n", " ")

        eval_prompt = self._render_evaluate_prompt(syllogism, translation)
        eval_response = await self._chat(eval_prompt)
        evaluation = self._parse_evaluation(eval_response)

        if not evaluation["correct"]:
            logger.info(f"  Evaluation: INCORRECT - {evaluation['feedback']}")
            logger.warning("  Retrying with feedback...")

            retry_prompt = self._render_translate_prompt(syllogism, feedback=evaluation["feedback"])
            translation = await self._chat(retry_prompt)
            translation = translation.replace("\n", " ")
            logger.debug(f"  Retry translation: {translation}")
        else:
            logger.info(f"  Evaluation: CORRECT")

        return translation

    async def _chat(self, prompt: str) -> str:
        response = await asyncio.wait_for(
            self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx,
                },
            ),
            timeout=self.timeout_seconds,
        )
        return response.message.content.replace("<|END_RESPONSE|>", "").strip()

    def _render_translate_prompt(self, syllogism: str, feedback: str | None = None) -> str:
        sentences = split_syllogism(syllogism)
        formatted_text = "\n".join(sentences)
        return self.translate_template.render(formatted_text=formatted_text, feedback=feedback)

    def _render_evaluate_prompt(self, original: str, translation: str) -> str:
        sentences = split_syllogism(original)
        formatted_original = "\n".join(sentences)
        return self.evaluate_template.render(formatted_original=formatted_original, translation=translation)

    @staticmethod
    def _parse_evaluation(response_text: str) -> dict:
        json_start = response_text.rfind("{")
        json_end = response_text.rfind("}") + 1

        if json_start != -1 and json_end > json_start:
            try:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                if "correct" in result and "feedback" in result:
                    return result
            except json.JSONDecodeError:
                pass

        return {"correct": True, "feedback": "Could not parse evaluation response, assuming correct."}
