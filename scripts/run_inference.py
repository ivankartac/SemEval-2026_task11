import argparse
import asyncio
import json
import logging
import re

import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jinja2 import Template

from functools import partial

from src.orchestrator import run_worker_queue
from src.pipeline.end_to_end import EndToEndPredictor
from src.pipeline.fol_parser import FOLParser
from src.pipeline.fol_prover import FOLProver, LLMProver
from src.pipeline.premise_retrieval import LLMPremiseRetrieval, SymbolicPremiseRetrieval
from src.pipeline.translator import Translator
from src.pipeline.transpiler import Transpiler
from src.utils import split_syllogism

logger = logging.getLogger(__name__)


async def process_example(
    example: dict,
    args,
    fol_parser: FOLParser | None,
    transpiler: Transpiler | None,
    prover: FOLProver,
    translator: Translator | None = None,
    llm_prover: LLMProver | None = None,
    llm_premise_retrieval: LLMPremiseRetrieval | None = None,
) -> dict | None:
    logger.info(
        f"Example {example['id']} (valid: {example.get('validity', 'N/A')}, plausible: {example.get('plausibility', 'N/A')})"
    )
    syllogism = example["syllogism"]

    # Translate syllogism if translator is provided and no translation exists yet
    if translator is not None and "syllogism_orig" not in example:
        try:
            translation = await translator(syllogism)
            example["syllogism_orig"] = syllogism
            syllogism = translation
        except Exception as e:
            logger.warning(f"Translation failed: {e}, using original syllogism")

    propositions = split_syllogism(syllogism)

    original_propositions = None
    if "syllogism_orig" in example:
        original_propositions = split_syllogism(example["syllogism_orig"])

        # Ensure length matches, otherwise ignore
        if len(original_propositions) != len(propositions):
            propositions = [p for p in propositions if p]
            original_propositions = [p for p in original_propositions if p]

        if len(original_propositions) != len(propositions):
            logger.warning(
                f"Sentence count mismatch for {example['id']}. Ignoring original props."
            )
            original_propositions = None

    if args.single_step:
        syllogism_text = syllogism if isinstance(syllogism, str) else ". ".join(syllogism)
        syllogism_orig_text = None
        if "syllogism_orig" in example:
            syllogism_orig = example["syllogism_orig"]
            syllogism_orig_text = syllogism_orig if isinstance(syllogism_orig, str) else ". ".join(syllogism_orig)
        parsed_propositions, full_responses, failed, metadata = await fol_parser.run_single_step(
            syllogism_text,
            syllogism_orig=syllogism_orig_text,
        )
    else:
        parsed_propositions, full_responses, failed, metadata = await fol_parser(
            propositions,
            original_propositions=original_propositions,
        )

    if failed:
        result_dict = {
            "id": example["id"],
            "valid": False,
            "syllogism": propositions,
            "llm_parsed": parsed_propositions,
            "llm_response": full_responses,
            "prover9": None,
            "had_retries": metadata["had_retries"],
            "had_invalid_format": metadata["had_invalid_format"],
            "had_additional_propositions": False,
            "relevant_premises": [],
        }
        if "attempts" in metadata:
            result_dict["attempts"] = metadata["attempts"]
        return result_dict

    if llm_prover is not None:
        result, llm_prover_response = await llm_prover(parsed_propositions)

        relevant_premise_indices = []
        if args.premise_retrieval == "llm" and result and llm_premise_retrieval is not None:
            relevant_premise_indices = await llm_premise_retrieval(propositions)

        return {
            "id": example["id"],
            "valid": result,
            "syllogism": propositions,
            "llm_parsed": parsed_propositions,
            "llm_response": full_responses,
            "llm_prover_response": llm_prover_response,
            "prover9": parsed_propositions,
            "had_retries": metadata["had_retries"],
            "had_invalid_format": metadata["had_invalid_format"],
            "had_additional_propositions": False,
            "relevant_premises": relevant_premise_indices,
        }

    # Validate propositions before translation
    if not parsed_propositions or len(parsed_propositions) < 2:
        logger.warning(
            f"Invalid parsed_propositions (empty or too short): {parsed_propositions}"
        )
        return {
            "id": example["id"],
            "valid": False,
            "syllogism": propositions,
            "llm_parsed": parsed_propositions,
            "llm_response": full_responses,
            "prover9": None,
            "had_retries": metadata["had_retries"],
            "had_invalid_format": True,
            "had_additional_propositions": False,
            "relevant_premises": [],
            **({"attempts": metadata["attempts"]} if "attempts" in metadata else {}),
        }
    if transpiler is not None:
        premises_raw, conclusion_raw, had_additional_propositions = transpiler(
            parsed_propositions
        )
        prover9_formatted = premises_raw + [conclusion_raw]
    else:
        # Direct prover syntax: clean up and use as-is
        parsed_propositions = [re.sub(r"\\(.)", r"\1", p) for p in parsed_propositions]
        parsed_propositions = [re.sub(r"(?<![A-Za-z]),|,(?!\s*[A-Za-z])", "", p) for p in parsed_propositions]
        parsed_propositions = [re.sub(r"[\+\\\?\!\*\\.\_\;]", "", p) for p in parsed_propositions]
        parsed_propositions = [re.sub(r"  +", " ", p).strip() for p in parsed_propositions]
        additional_propositions = Transpiler.get_additional_propositions(parsed_propositions)
        premises_raw = parsed_propositions[:-1] + additional_propositions
        conclusion_raw = parsed_propositions[-1]
        prover9_formatted = parsed_propositions
        had_additional_propositions = len(additional_propositions) > 0

    prover_result = prover(premises_raw, conclusion_raw)
    result = prover_result.valid

    relevant_premise_indices = []
    if args.premise_retrieval == "llm" and result and llm_premise_retrieval is not None:
        relevant_premise_indices = await llm_premise_retrieval(propositions)
    elif args.premise_retrieval == "symbolic" and result:
        # Use min to handle cases where parsed propositions don't match input count
        num_premises_to_check = min(len(propositions) - 1, len(parsed_propositions) - 1, len(prover_result.premises))
        relevant_premise_indices = SymbolicPremiseRetrieval()(
            prover_result.premises[:num_premises_to_check], prover_result.conclusion
        )

    result_dict = {
        "id": example["id"],
        "valid": result,
        "syllogism": propositions,
        "syllogism_orig": (original_propositions if original_propositions else []),
        "llm_parsed": parsed_propositions,
        "llm_response": full_responses,
        "prover9": prover9_formatted,
        "had_retries": metadata["had_retries"],
        "had_invalid_format": metadata["had_invalid_format"],
        "had_additional_propositions": had_additional_propositions,
        "relevant_premises": relevant_premise_indices,
    }

    if "attempts" in metadata:
        result_dict["attempts"] = metadata["attempts"]

    return result_dict


async def main(args):
    script_dir = Path(__file__).parent.parent

    if not args.end_to_end:
        if args.parser_syntax == "latex":
            template_initial_path = script_dir / "prompt_templates" / "fol_parser_initial.jinja"
            template_default_path = script_dir / "prompt_templates" / "fol_parser_default.jinja"
        else:
            template_initial_path = script_dir / "prompt_templates" / "fol_parser_prover_syntax_initial.jinja"
            template_default_path = script_dir / "prompt_templates" / "fol_parser_prover_syntax_default.jinja"
        template_single_step_path = script_dir / "prompt_templates" / "fol_parser_single_step.jinja"
        template_prover_path = (
            script_dir / "prompt_templates" / "llm_prover.jinja"
        )
        template_premise_retrieval_path = (
            script_dir / "prompt_templates" / "llm_premise_retrieval.jinja"
        )

    data_path = Path(args.input)

    with open(data_path, "r") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} examples.")

    api_urls = args.ollama_url
    num_gpus = len(api_urls)

    if args.end_to_end:
        if args.premise_retrieval:
            template_e2e_path = script_dir / "prompt_templates" / "end_to_end_premise_retrieval.jinja"
        else:
            template_e2e_path = script_dir / "prompt_templates" / "end_to_end_baseline.jinja"
        with open(template_e2e_path, "r") as f:
            end_to_end_template = Template(f.read())

        end_to_end_predictors = []
        for url in api_urls:
            end_to_end_predictors.append(
                EndToEndPredictor(
                    model=args.model,
                    template=end_to_end_template,
                    api_url=url,
                    timeout_seconds=args.timeout,
                    num_ctx=args.num_ctx,
                    num_predict=args.num_predict,
                )
            )

        fol_parsers = []
        transpiler = None
        llm_provers = [None] * num_gpus
        llm_premise_retrievals = [None] * num_gpus
        translators = [None] * num_gpus
        prover = None
    else:
        end_to_end_predictors = [None] * num_gpus

        with open(template_initial_path, "r") as f:
            prompt_template_initial = Template(f.read())

        with open(template_default_path, "r") as f:
            prompt_template_default = Template(f.read())

        with open(template_prover_path, "r") as f:
            prompt_template_prover = Template(f.read())

        with open(template_single_step_path, "r") as f:
            prompt_template_single_step = Template(f.read())

        if args.premise_retrieval == "llm":
            with open(template_premise_retrieval_path, "r") as f:
                premise_retrieval_template = Template(f.read())
        else:
            premise_retrieval_template = None

        prover = FOLProver()

        fol_parsers = []
        for url in api_urls:
                fol_parsers.append(
                    FOLParser(
                        args.model,
                        prompt_template_default,
                        prompt_template_initial,
                        template_single_step=prompt_template_single_step,
                        api_url=url,
                        timeout_seconds=args.timeout,
                        num_ctx=args.num_ctx,
                        num_predict=args.num_predict,
                    )
                )
        transpiler = Transpiler(normalize_predicates=False) if args.parser_syntax == "latex" else None

        # Create per-GPU LLMProver instances
        if args.prover == "llm":
            llm_provers = []
            for url in api_urls:
                llm_provers.append(
                    LLMProver(
                        model=args.model,
                        template=prompt_template_prover,
                        api_url=url,
                        timeout_seconds=args.timeout,
                        num_ctx=args.num_ctx,
                        num_predict=args.num_predict,
                    )
                )
        else:
            llm_provers = [None] * num_gpus

        # Create per-GPU LLMPremiseRetrieval instances
        if args.premise_retrieval == "llm":
            llm_premise_retrievals = []
            for url in api_urls:
                llm_premise_retrievals.append(
                    LLMPremiseRetrieval(
                        model=args.model,
                        template=premise_retrieval_template,
                        api_url=url,
                        timeout=args.timeout,
                        num_ctx=args.num_ctx,
                        num_predict=args.num_predict,
                    )
                )
        else:
            llm_premise_retrievals = [None] * num_gpus

        # Create per-GPU Translator instances
        if args.translation_model:
            with open(script_dir / "prompt_templates" / "translator.jinja", "r") as f:
                translate_template = Template(f.read())
            with open(script_dir / "prompt_templates" / "translator_evaluate.jinja", "r") as f:
                translate_evaluate_template = Template(f.read())
            translators = []
            for url in api_urls:
                translators.append(
                    Translator(
                        model=args.translation_model,
                        translate_template=translate_template,
                        evaluate_template=translate_evaluate_template,
                        api_url=url,
                        num_ctx=args.translator_num_ctx,
                        timeout_seconds=args.timeout,
                    )
                )
        else:
            translators = [None] * num_gpus

    # Build per-GPU processing functions
    process_fns = []
    for gpu_idx in range(num_gpus):
        if args.end_to_end:
            process_fns.append(end_to_end_predictors[gpu_idx])
        else:
            parser = fol_parsers[gpu_idx] if fol_parsers else None
            process_fns.append(
                partial(
                    process_example,
                    args=args,
                    fol_parser=parser,
                    transpiler=transpiler,
                    prover=prover,
                    translator=translators[gpu_idx],
                    llm_prover=llm_provers[gpu_idx],
                    llm_premise_retrieval=llm_premise_retrievals[gpu_idx],
                )
            )

    await run_worker_queue(
        data,
        process_fns,
        output_path=args.output,
        batch_size=args.batch_size,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=str,
        help="Input JSON file",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="YAML config file (CLI arguments override config values)",
    )
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument(
        "--ollama-url",
        type=str,
        nargs="+",
        help="Ollama server URL(s). Multiple URLs for multi-GPU inference.",
    )
    parser.add_argument(
        "--translation-model",
        type=str,
        help="Translate syllogisms to English before processing, using this model",
    )
    parser.add_argument(
        "--parser-syntax",
        type=str,
        choices=["latex", "prover9"],
        help="FOL syntax the parser outputs: 'latex' (transpiled to Prover9) or 'prover9' (used directly)",
    )
    parser.add_argument(
        "--prover",
        type=str,
        choices=["symbolic", "llm"],
        help="Prover to use: 'symbolic' uses Prover9, 'llm' uses an LLM",
    )
    parser.add_argument(
        "--premise-retrieval",
        type=str,
        choices=["symbolic", "llm"],
        help="Identify relevant premises: 'symbolic' uses Prover9, 'llm' uses an LLM",
    )
    parser.add_argument(
        "--end-to-end",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply LLM directly to input syllogism to predict validity (no parsing/proving)",
    )
    parser.add_argument(
        "--single-step",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Parse all propositions to FOL in a single inference call (instead of one at a time)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Number of examples to process in parallel (requires OLLAMA_NUM_PARALLEL to be set accordingly)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout in seconds for each LLM request (default: 300, increase for thinking models)",
    )
    parser.add_argument("--num-ctx", type=int, help="Context window size (default: 16384)")
    parser.add_argument("--num-predict", type=int, help="Max output tokens (default: 16384)")
    parser.add_argument("--translator-num-ctx", type=int, help="Context window size for translator (default: 4096)")
    args = parser.parse_args()

    # Load config file defaults
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}

    # Apply defaults: config values fill in for unset CLI args
    defaults = {
        "model": "qwen3:4b-thinking-2507-fp16",
        "output": None,
        "ollama_url": ["http://localhost:11434"],
        "end_to_end": False,
        "translation_model": None,
        "parser_syntax": "latex",
        "single_step": False,
        "prover": "symbolic",
        "premise_retrieval": None,
        "batch_size": 1,
        "timeout": 300,
        "num_ctx": 16384,
        "num_predict": 16384,
        "translator_num_ctx": 4096,
    }

    for key, default_value in defaults.items():
        cli_value = getattr(args, key, None)
        config_value = config.get(key, None)
        if cli_value is not None:
            setattr(args, key, cli_value)
        elif config_value is not None:
            setattr(args, key, config_value)
        else:
            setattr(args, key, default_value)

    return args


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    asyncio.run(main(args))
