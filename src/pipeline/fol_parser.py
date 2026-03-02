import asyncio
import json
import logging
import re

import ollama
from jinja2 import Template

from src.utils import split_syllogism

logger = logging.getLogger(__name__)

RE_BOX = re.compile(r"\\boxed\{([^}]+)\}")


class InvalidFormatError(Exception):
    pass


class FOLParser:
    def __init__(
        self,
        model: str,
        template_default: Template,
        template_initial: Template,
        template_single_step: Template = None,
        api_url: str = None,
        retry_attempts: int = 3,
        timeout_seconds: int = 120,
        num_ctx: int = 16384,
        num_predict: int = 16384,
    ):
        self.model = model
        self.template_default = template_default
        self.template_initial = template_initial
        self.template_single_step = template_single_step
        self.num_ctx = num_ctx
        self.num_predict = num_predict

        if api_url:
            self.client = ollama.AsyncClient(host=api_url)
        else:
            self.client = ollama.AsyncClient()

        self.retry_attempts = retry_attempts
        self.timeout_seconds = timeout_seconds

    async def __call__(
        self,
        propositions: list[str],
        original_propositions: list[str] = None,
    ) -> tuple[list[str], list[str], bool, dict]:
        previous_propositions = []
        full_responses = []
        failed = False
        metadata = {
            "had_retries": False,
            "had_invalid_format": False,
        }

        for i, proposition in enumerate(propositions):
            orig_prop = (
                original_propositions[i]
                if original_propositions and i < len(original_propositions)
                else None
            )
            prompt = self._render_prompt(
                proposition,
                previous_propositions,
                propositions[:i],
                original_statement=orig_prop,
            )

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

                    pattern = r"\\boxed\{([^}]+)\}"
                    matches = re.findall(pattern, content)

                    if not matches:
                        metadata["had_invalid_format"] = True
                        raise InvalidFormatError("Invalid response format")

                    # Track if we had to retry
                    if attempt > 0:
                        metadata["had_retries"] = True
                    break

                except asyncio.TimeoutError:
                    if attempt < self.retry_attempts - 1:
                        metadata["had_retries"] = True
                        logger.warning(f"Attempt {attempt + 1} timed out. Retrying...")
                    else:
                        logger.error(f"All {self.retry_attempts} attempts timed out.")
                        failed = True
                except Exception as e:
                    if attempt < self.retry_attempts - 1:
                        metadata["had_retries"] = True
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    else:
                        logger.error(f"All {self.retry_attempts} attempts failed.")
                        failed = True

            if failed:
                # Stop processing further propositions for this example
                break

            content = content.strip()
            fol_proposition = self._extract_fol_proposition(content)
            logger.debug(f"{proposition}: {fol_proposition}")
            previous_propositions.append(fol_proposition)
            full_responses.append(content)

        return previous_propositions, full_responses, failed, metadata

    async def run_single_step(
        self,
        syllogism: str,
        syllogism_orig: str = None,
    ) -> tuple[list[str], str, bool, dict]:
        """Translate all propositions to FOL in a single inference call.

        Args:
            syllogism: The full syllogism text (premises and conclusion).
            syllogism_orig: The original syllogism text (for multilingual templates).

        Returns:
            tuple containing:
            - parsed_propositions: list of FOL propositions
            - full_response: the full response from the model
            - failed: bool indicating if the parsing failed
            - metadata: dict with parsing metadata

        """
        metadata = {
            "had_retries": False,
            "had_invalid_format": False,
        }

        attempts = []

        syllogism = "\n".join(split_syllogism(syllogism))
        num_premises = len(syllogism.split("\n")) - 1
        num_premises = self._number_to_word(num_premises)
        if syllogism_orig:
            syllogism_orig = "\n".join(split_syllogism(syllogism_orig))
            prompt = self.template_single_step.render(syllogism=syllogism, syllogism_orig=syllogism_orig, num_premises=num_premises)
        else:
            prompt = self.template_single_step.render(syllogism=syllogism, num_premises=num_premises)

        for attempt in range(self.retry_attempts):
            temperature = 0.6 if attempt > 0 else 0.0
            content = None

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

                logger.debug(f"Single-step response:\n\n{content}")

                content_list = self._extract_json_array(content)
                if content_list is None:
                    metadata["had_invalid_format"] = True
                    attempts.append(content)
                    raise InvalidFormatError(
                        "Could not extract JSON array from single-step response"
                    )

                # Validate that the number of FOL formulas matches the number of input propositions
                num_propositions = len(syllogism.split("\n"))
                if len(content_list) != num_propositions:
                    metadata["had_invalid_format"] = True
                    attempts.append(content)
                    raise InvalidFormatError(
                        f"Number of FOL formulas ({len(content_list)}) does not match "
                        f"number of input propositions ({num_propositions})"
                    )

                for item in content_list:
                    proposition = item["proposition"]
                    fol_proposition = item["fol_formula"]
                    logger.debug(f"{proposition}: {fol_proposition}")

                if attempt > 0:
                    metadata["had_retries"] = True

                parsed_propositions = [item["fol_formula"] for item in content_list]
                if attempts:
                    metadata["attempts"] = attempts
                return parsed_propositions, content, False, metadata

            except asyncio.TimeoutError:
                attempts.append({"error": "timeout", "attempt": attempt + 1})
                if attempt < self.retry_attempts - 1:
                    metadata["had_retries"] = True
                    logger.warning(f"Single-step attempt {attempt + 1} timed out. Retrying...")
                else:
                    logger.error(f"Single-step: all {self.retry_attempts} attempts timed out.")
            except Exception as e:
                if content is not None and content not in attempts:
                    attempts.append(content)
                if attempt < self.retry_attempts - 1:
                    metadata["had_retries"] = True
                    logger.warning(f"Single-step attempt {attempt + 1} failed: {e}. Retrying...")
                else:
                    logger.error(f"Single-step: all {self.retry_attempts} attempts failed.")

        # All attempts failed
        metadata["attempts"] = attempts
        return [], "", True, metadata

    def _number_to_word(self, number: int) -> str:
        if number == 1:
            return "one"
        elif number == 2:
            return "two"
        elif number == 3:
            return "three"
        elif number == 4:
            return "four"
        elif number == 5:
            return "five"
        elif number == 6:
            return "six"
        elif number == 7:
            return "seven"
        elif number == 8:
            return "eight"
        elif number == 9:
            return "nine"
        elif number == 10:
            return "ten"
        else:
            return str(number)

    def _render_prompt(
        self,
        proposition: str,
        previous_propositions: list[str],
        original_propositions: list[str],
        original_statement: str = None,
    ) -> str:
        if not previous_propositions:
            prompt = self.template_initial.render(
                statement=proposition, statement_orig=original_statement
            )
        else:
            previous_propositions_formatted = "\n".join(
                [
                    f"{i}. {s}: {t}"
                    for i, (s, t) in enumerate(
                        zip(original_propositions, previous_propositions), 1
                    )
                ]
            )
            prompt = self.template_default.render(
                statement=proposition,
                previous_statements=previous_propositions_formatted,
                statement_orig=original_statement,
            )
        return prompt

    @staticmethod
    def _extract_fol_proposition(response: str) -> str:
        if "</think>" in response:
            response = response.split("</think>")[-1]
        response = re.sub(r"\\?\\text\{([^}]+)\}", r"\1", rf"{response}")
        matches = re.findall(RE_BOX, response)
        result = matches[-1]
        return rf"{result}"

    @staticmethod
    def _extract_json_array(text: str) -> list | None:
        text = text.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        if "</think>" in text:
            text = text.split("</think>")[-1].strip()

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code blocks (```json ... ``` or ``` ... ```)
        code_block_patterns = [r"```json\s*(.*?)```", r"```\s*(.*?)```"]
        for pattern in code_block_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(1).strip())
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    pass

        # Find JSON array patterns ([\s*{) and try the LAST one first
        # This handles cases where text contains brackets like [since...]
        json_array_starts = [m.start() for m in re.finditer(r'\[\s*[\{\n]', text)]

        # Try from last to first (most likely the actual JSON is at the end)
        for start in reversed(json_array_starts):
            depth = 0
            for i, char in enumerate(text[start:], start):
                if char == "[":
                    depth += 1
                elif char == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            result = json.loads(text[start : i + 1])
                            if isinstance(result, list):
                                return result
                        except json.JSONDecodeError:
                            pass
                        break

        # Fallback: try the first [ if no JSON array pattern found
        start = text.find("[")
        if start != -1:
            depth = 0
            for i, char in enumerate(text[start:], start):
                if char == "[":
                    depth += 1
                elif char == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            result = json.loads(text[start : i + 1])
                            if isinstance(result, list):
                                return result
                        except json.JSONDecodeError:
                            pass
                        break

        # Final fallback: handle consecutive JSON objects without array wrapper
        # Some models output: {...} {...} {...} instead of [{...}, {...}, {...}]
        objects = []
        depth = 0
        obj_start = None

        for i, char in enumerate(text):
            if char == '{':
                if depth == 0:
                    obj_start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and obj_start is not None:
                    try:
                        obj = json.loads(text[obj_start : i + 1])
                        objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    obj_start = None

        if objects and all('proposition' in obj and 'fol_formula' in obj for obj in objects):
            return objects

        return None
