import asyncio
import json
import logging
import re

import ollama
from jinja2 import Template
from nltk.inference.prover9 import Prover9

from src.pipeline.fol_parser import InvalidFormatError

logger = logging.getLogger(__name__)


class SymbolicPremiseRetrieval:
    """Identify relevant premises by checking which are needed for the proof.

    Iterates over premises and checks if removing each one breaks the proof.
    If the proof still holds without a premise, it is not relevant.
    """

    def __call__(self, premises, conclusion) -> list[int]:
        """
        Args:
            premises: List of NLTK Expression objects (parsed FOL premises).
            conclusion: NLTK Expression object (parsed FOL conclusion).

        Returns:
            List of 0-based indices of relevant premises.
        """
        relevant_premises = []
        relevant_premise_indices = []
        current_premises = list(premises)

        for i in range(len(current_premises)):
            left_out_premise = current_premises.pop(0)
            if not Prover9().prove(conclusion, relevant_premises + current_premises):
                relevant_premises.append(left_out_premise)
                relevant_premise_indices.append(i)

        return relevant_premise_indices


class LLMPremiseRetrieval:
    def __init__(
        self,
        model: str,
        template: Template,
        api_url: str = None,
        timeout: int = 300,
        retry_attempts: int = 3,
        num_ctx: int = 16384,
        num_predict: int = 16384,
    ):
        self.model = model
        self.template = template
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.num_ctx = num_ctx
        self.num_predict = num_predict

        if api_url:
            self.client = ollama.AsyncClient(host=api_url)
        else:
            self.client = ollama.AsyncClient()

    async def __call__(self, propositions: list[str]) -> list[int]:
        """Identify relevant premises using an LLM.

        Args:
            propositions: List of natural language propositions (premises + conclusion).

        Returns:
            List of 0-based indices of relevant premises.
        """
        premises = propositions[:-1]
        conclusion = propositions[-1]

        premises_formatted = "\n".join(
            f"{i}. {p}" for i, p in enumerate(premises)
        )
        prompt = self.template.render(premises=premises_formatted, conclusion=conclusion)

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
                    timeout=self.timeout,
                )
                content = response.message.content.strip()

                if "</think>" in content:
                    content = content.split("</think>")[-1].strip()

                # Extract JSON array of integers
                match = re.search(r"\[[\d\s,]*\]", content)
                if not match:
                    raise InvalidFormatError(
                        f"Could not extract JSON array of indices from response: {content}"
                    )

                indices = json.loads(match.group())
                if not all(isinstance(i, int) and 0 <= i < len(premises) for i in indices):
                    raise InvalidFormatError(
                        f"Invalid indices in response: {indices}"
                    )

                logger.info(f"LLM identified relevant premises: {indices}")
                return sorted(set(indices))

            except asyncio.TimeoutError:
                if attempt < self.retry_attempts - 1:
                    logger.warning(f"Identify-relevant-llm attempt {attempt + 1} timed out. Retrying...")
                else:
                    logger.error(f"Identify-relevant-llm: all {self.retry_attempts} attempts timed out.")
            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    logger.warning(f"Identify-relevant-llm attempt {attempt + 1} failed: {e}. Retrying...")
                else:
                    logger.error(f"Identify-relevant-llm: all {self.retry_attempts} attempts failed: {e}")

        return []
