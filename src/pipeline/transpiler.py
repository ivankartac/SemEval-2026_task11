import logging
import re

logger = logging.getLogger(__name__)


class Transpiler:
    def __init__(self, target_syntax: str | None = "prover9", normalize_predicates: bool = True):
        if target_syntax not in ["prover9", None]:
            raise ValueError(f"Invalid target syntax: {target_syntax}")
        self.target_syntax = target_syntax
        self.normalize_predicates = normalize_predicates

    def __call__(self, propositions: list[str]) -> tuple[list[str], str, bool]:
        target_propositions = []
        logger.info(f"Transpiling to {self.target_syntax} syntax")
        for p_orig in propositions:
            # TODO: add other operators not present in training data?
            p = p_orig
            p = re.sub(r"\\text\{([^}]+)\}", r"\1", p)
            p = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", p)
            p = p.replace(r"\ ", "")
            p = p.replace(r"\exists", "exists")
            p = p.replace(r"\forall", "all")
            p = p.replace(r"\neg ", "-")
            p = p.replace(r"\land", "&")
            p = p.replace(r"\lor", "|")
            p = p.replace(r"\rightarrow", "->")
            p = p.replace(r"\leftarrow", "<-")
            p = p.replace(r"\leftrightarrow", "<->")
            p = p.replace(r"\to", "->")
            p = p.replace(r"\left", "")
            p = p.replace(r"\right", "")
            p = p.replace(r"\big", "")
            p = p.replace(r"\neq", "!=")
            p = p.replace(r"\ne", "!=")

            # Remove commas except when between variables (e.g., "x, y", "F, y" or "x,y")
            p = re.sub(r"(?<![A-Za-z]),|,(?!\s*[A-Za-z])", "", p)
            # Remove noise (invalid characters)
            p = re.sub(r"[\+\\\?\!\*\\.\_]", "", p)
            # Remove predicates with empty arguments (e.g., "P()" -> "")
            p = re.sub(r"-?[A-Za-z]+\(\)\s*(&|(\|)|(->))?\s*", "", p)
            # Normalize multiple spaces to single space
            p = re.sub(r"  +", " ", p)

            # Parethesize universal/existential statements
            if "exists" in p:
                p = f"({p})"
            elif "all" in p:
                p = f"({p})"
            # Parethesize predicates
            p = re.sub(
                r"exists \(?([a-z])\)? (-?[A-Za-z]+\([A-Za-z]+(,\s*[A-Za-z]+)*\))",
                r"exists \1 (\2)",
                p,
            )
            p = re.sub(
                r"all \(?([a-z])\)? (-?[A-Za-z]+\([A-Za-z]+(,\s*[A-Za-z]+)*\))",
                r"all \1 (\2)",
                p,
            )

            logger.debug(f"{p_orig}: {p}")
            target_propositions.append(p)

        # Uppercase single-character predicates (e.g., "a(x)" -> "A(x)")
        for i, p in enumerate(target_propositions):
            target_propositions[i] = re.sub(
                r"\b([a-z])\(", lambda m: m.group(1).upper() + "(", p
            )

        # Lowercase predicate arguments (e.g., "R(B, A)" -> "R(b, a)")
        # This prevents arity conflicts where a letter is used as both predicate and argument
        def lowercase_args(match):
            predicate = match.group(1)
            args = match.group(2)
            arg_list = [a.strip() for a in args.split(',')]

            processed_args = []
            for arg in arg_list:
                arg_lower = arg.lower()
                # If argument is more than 1 character, it's a full word - append "argument" to avoid collisions with predicate names
                if len(arg_lower) > 1:
                    arg_lower = arg_lower + "argument"
                processed_args.append(arg_lower)
            return predicate + "(" + ", ".join(processed_args) + ")"

        for i, p in enumerate(target_propositions):
            target_propositions[i] = re.sub(
                r"([A-Za-z]+)\(([^)]+)\)",
                lowercase_args,
                p
            )

        # Normalize predicate names (e.g., "Car" and "C" -> "C")
        if self.normalize_predicates:
            # Find all predicate names across all propositions
            all_predicates = set()
            for p in target_propositions:
                predicates = re.findall(r"([A-Za-z]+)\([A-Za-z]", p)
                all_predicates.update(predicates)

            # Build mapping from longer predicates to shorter ones
            # Exception: don't normalize predicates that are 1-2 characters (e.g., "Sh" stays "Sh")
            predicate_mapping = {}
            for pred in all_predicates:
                if len(pred) <= 2:
                    continue
                for other in all_predicates:
                    if pred != other and pred.startswith(other) and len(other) < len(pred):
                        if pred not in predicate_mapping or len(other) < len(
                            predicate_mapping[pred]
                        ):
                            predicate_mapping[pred] = other

            # Apply normalization
            if predicate_mapping:
                logger.info(f"Normalizing predicates: {predicate_mapping}")
                for i, p in enumerate(target_propositions):
                    for long_pred, short_pred in predicate_mapping.items():
                        p = re.sub(rf"\b{long_pred}\(", f"{short_pred}(", p)
                    target_propositions[i] = p

        # Add existential statements required by the prover for the conclusion to be valid
        additional_propositions = Transpiler.get_additional_propositions(target_propositions)

        had_additional_propositions = len(additional_propositions) > 0

        premises_raw = target_propositions[:-1] + additional_propositions
        conclusion_raw = target_propositions[-1]

        return premises_raw, conclusion_raw, had_additional_propositions

    @staticmethod
    def get_additional_propositions(target_propositions):
        additional_propositions = []
        if target_propositions[-1].startswith("(exists x") or target_propositions[-1].startswith("exists x"):
            for p in target_propositions[:-1]:
                if not p.startswith("(all x") and not p.startswith("all x"):
                    continue
                predicates = re.findall(r"[A-Za-z]+\(x\)", p)
                for predicate in predicates:
                    existential_predicate = predicate.replace("x", "y")
                    additional_propositions.append(
                        f"(exists y ({existential_predicate}))"
                    )
            predicates = re.findall(r"[A-Za-z]+\(x\)", target_propositions[-1])
            if len(predicates) > 1:
                for predicate in predicates:
                    existential_predicate = predicate.replace("x", "y")
                    additional_propositions.append(
                        f"(exists y ({existential_predicate}))"
                    )
        additional_propositions = list(set(additional_propositions))

        for p in additional_propositions:
            logger.debug(f"Additional propositions: {p}")

        return additional_propositions
