import asyncio
import logging
import re
from dataclasses import dataclass

import ollama
from jinja2 import Template
from nltk.inference.prover9 import Prover9
from nltk.sem import Expression

from src.pipeline.fol_parser import InvalidFormatError

logger = logging.getLogger(__name__)


@dataclass
class ProverResult:
    valid: bool
    premises: list  # NLTK Expression objects (for downstream use e.g. premise retrieval)
    conclusion: object  # NLTK Expression object


class FOLProver:
    def __init__(self):
        self.read_expr = Expression.fromstring

    def __call__(self, premises_raw: list[str], conclusion_raw: str) -> ProverResult:
        try:
            premises = [self.read_expr(x) for x in premises_raw]
            conclusion = self.read_expr(conclusion_raw)
            valid = Prover9().prove(conclusion, premises)
        except Exception as e:
            logger.error(f"{e}")
            logger.warning("Using fallback to false")
            return ProverResult(valid=False, premises=[], conclusion=None)

        logger.info(f"valid: {valid}")

        return ProverResult(valid=valid, premises=premises, conclusion=conclusion)


class LLMProver:
    def __init__(
        self,
        model: str,
        template: Template,
        api_url: str = None,
        retry_attempts: int = 3,
        timeout_seconds: int = 120,
        num_ctx: int = 16384,
        num_predict: int = 16384,
    ):
        self.model = model
        self.template = template
        self.retry_attempts = retry_attempts
        self.timeout_seconds = timeout_seconds
        self.num_ctx = num_ctx
        self.num_predict = num_predict

        if api_url:
            self.client = ollama.AsyncClient(host=api_url)
        else:
            self.client = ollama.AsyncClient()

    async def __call__(
        self,
        parsed_propositions: list[str],
    ) -> tuple[bool | None, str]:
        prompt = self.template.render(
            premises=parsed_propositions[:-1],
            conclusion=parsed_propositions[-1],
        )
        logger.debug(f"LLM Prover prompt:\n{prompt}")

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
                content = response.message.content.strip()

                logger.debug(f"LLM Prover response:\n\n{content}")

                boxed_match = re.search(r"\\boxed\{(\w+)\}", content)
                if boxed_match is None:
                    raise InvalidFormatError(
                        "Could not extract \\boxed{} from LLM Prover response"
                    )

                result = boxed_match.group(1).lower()
                result = True if result == "true" else False
                return result, content

            except asyncio.TimeoutError:
                if attempt < self.retry_attempts - 1:
                    logger.warning(f"LLM Prover attempt {attempt + 1} timed out. Retrying...")
                else:
                    logger.error(f"LLM Prover: all {self.retry_attempts} attempts timed out.")
            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    logger.warning(f"LLM Prover attempt {attempt + 1} failed: {e}. Retrying...")
                else:
                    logger.error(f"LLM Prover: all {self.retry_attempts} attempts failed.")

        return None, ""
