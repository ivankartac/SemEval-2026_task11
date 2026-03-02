"""
Evaluation code from the shared task's repo
https://github.com/neuro-symbolic-ai/semeval_2026_task_11/blob/main/evaluation_kit/task%202%20%26%204/evaluation_script.py
"""

import json
import math
import sys
from typing import Any, Dict, List, Optional, Tuple


def calculate_f1_premises(
    gt_map: Dict[str, Any], predictions: List[Dict[str, Any]]
) -> float:
    """
    Calculates the Macro-Averaged F1-Score for premise retrieval across all data points.
    """
    total_precision = 0.0
    total_recall = 0.0
    valid_count = 0

    for pred_item in predictions:
        item_id = pred_item["id"]
        if (
            item_id in gt_map
            and "relevant_premises" in gt_map[item_id]
            and "relevant_premises" in pred_item
        ):
            true_positives = set(gt_map[item_id]["relevant_premises"])
            predicted_positives = set(pred_item["relevant_premises"])

            if len(true_positives) == 0:
                continue

            TP = len(true_positives.intersection(predicted_positives))
            FP = len(predicted_positives.difference(true_positives))
            FN = len(true_positives.difference(predicted_positives))

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

            total_precision += precision
            total_recall += recall
            valid_count += 1

    if valid_count == 0:
        return 0.0

    macro_precision = total_precision / valid_count
    macro_recall = total_recall / valid_count

    f1_score = (
        2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
        if (macro_precision + macro_recall) > 0
        else 0.0
    )

    return f1_score * 100


def calculate_accuracy(
    ground_truth_list: List[Dict[str, Any]],
    predictions_list: List[Dict[str, Any]],
    metric_name: str,
    prediction_key: str,
    plausibility_filter: Optional[bool] = None,
) -> Tuple[float, int, int]:
    """
    Calculates the accuracy of 'validity' predictions against ground truth labels,
    with an optional filter based on ground truth 'plausibility'.
    """
    gt_map = {item["id"]: item for item in ground_truth_list}

    correct_predictions = 0
    total_predictions = 0

    for pred_item in predictions_list:
        item_id = pred_item["id"]

        if item_id in gt_map:
            gt_item = gt_map[item_id]

            gt_plausibility = gt_item.get("plausibility")
            if (
                plausibility_filter is not None
                and gt_plausibility != plausibility_filter
            ):
                continue

            if metric_name in gt_item and prediction_key in pred_item:
                true_label = gt_item[metric_name]
                predicted_label = pred_item[prediction_key]

                if isinstance(true_label, bool) and isinstance(predicted_label, bool):
                    total_predictions += 1
                    if true_label == predicted_label:
                        correct_predictions += 1

    if total_predictions == 0:
        return 0.0, 0, 0

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy, correct_predictions, total_predictions


def calculate_subgroup_accuracy(
    gt_map: Dict[str, Any],
    predictions_list: List[Dict[str, Any]],
    gt_validity: bool,
    gt_plausibility: bool,
) -> Tuple[float, int, int]:
    """
    Calculates accuracy for a specific subgroup defined by ground truth validity AND plausibility.
    """
    correct_predictions = 0
    total_predictions = 0

    for pred_item in predictions_list:
        item_id = pred_item["id"]
        if item_id in gt_map:
            gt_item = gt_map[item_id]

            if (
                gt_item.get("validity") == gt_validity
                and gt_item.get("plausibility") == gt_plausibility
            ):
                if "validity" in gt_item and "validity" in pred_item:
                    true_label = gt_item["validity"]
                    predicted_label = pred_item["validity"]

                    if isinstance(true_label, bool) and isinstance(
                        predicted_label, bool
                    ):
                        total_predictions += 1
                        if true_label == predicted_label:
                            correct_predictions += 1

    if total_predictions == 0:
        return 0.0, 0, 0

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy, correct_predictions, total_predictions


def calculate_content_effect_bias(accuracies: Dict[str, float]) -> Dict[str, float]:
    """
    Calculates the content effect bias metrics as defined.
    """

    acc_plausible_valid = accuracies.get("acc_plausible_valid", 0.0)
    acc_implausible_valid = accuracies.get("acc_implausible_valid", 0.0)
    acc_plausible_invalid = accuracies.get("acc_plausible_invalid", 0.0)
    acc_implausible_invalid = accuracies.get("acc_implausible_invalid", 0.0)

    # 1. content_effect_intra_validity_label
    intra_valid_diff = abs(acc_plausible_valid - acc_implausible_valid)
    intra_invalid_diff = abs(acc_plausible_invalid - acc_implausible_invalid)
    content_effect_intra_validity_label = (intra_valid_diff + intra_invalid_diff) / 2.0

    # 2. content_effect_inter_validity_label
    inter_plausible_diff = abs(acc_plausible_valid - acc_plausible_invalid)
    inter_implausible_diff = abs(acc_implausible_valid - acc_implausible_invalid)
    content_effect_inter_validity_label = (
        inter_plausible_diff + inter_implausible_diff
    ) / 2.0

    # 3. tot_content_effect
    tot_content_effect = (
        content_effect_intra_validity_label + content_effect_inter_validity_label
    ) / 2.0

    return {
        "content_effect_intra_validity_label": content_effect_intra_validity_label,
        "content_effect_inter_validity_label": content_effect_inter_validity_label,
        "tot_content_effect": tot_content_effect,
    }


def calculate_smooth_combined_metric(
    overall_metric: float, total_content_effect: float
) -> float:
    """
    Computes the smoother combined score using the natural logarithm.
    Formula: overall_metric / (1 + ln(1 + content_effect))
    """
    if total_content_effect < 0:
        return 0.0

    log_penalty = math.log(1 + total_content_effect)
    smoother_score = overall_metric / (1 + log_penalty)
    return smoother_score


def run_full_scoring(
    reference_data_path: str,
    prediction_path: str,
    output_path: str,
    verbose: bool = True,
):
    """
    Runs the full analysis pipeline for a single submission and writes results to the output path.
    """

    try:
        # 1. Load data from the provided file paths
        with open(reference_data_path, "r") as f:
            ground_truth = json.load(f)

        with open(prediction_path, "r") as f:
            predictions = json.load(f)

        # Check that the examples in ground_truth are all covered in predictions
        gt_ids = set([example["id"] for example in ground_truth])
        pd_ids = set([example["id"] for example in predictions])
        diff = len(gt_ids.difference(pd_ids))

        if diff != 0:
            print(
                f"Error: not all the examples in ground truth have a corresponding prediction",
                file=sys.stderr,
            )
            final_results = {"accuracy": 0.0, "f1_premises": 0.0, "combined_score": 0.0}
            # --- Write Results ---
            try:
                with open(output_path, "w") as f:
                    json.dump(final_results, f)
            except Exception as e:
                print(f"Error writing final results to file: {e}", file=sys.stderr)

            return

    except FileNotFoundError as e:
        print(f"Error: Required file not found. {e}", file=sys.stderr)
        final_results = {
            "accuracy": 0.0,
            "f1_premises": 0.0,
            "content_effect": 100.0,
            "combined_score": 0.0,
        }
    except json.JSONDecodeError as e:
        print(
            f"Error: Invalid JSON format in submission or reference file. {e}",
            file=sys.stderr,
        )
        final_results = {
            "accuracy": 0.0,
            "f1_premises": 0.0,
            "content_effect": 100.0,
            "combined_score": 0.0,
        }
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}", file=sys.stderr)
        final_results = {
            "accuracy": 0.0,
            "f1_premises": 0.0,
            "content_effect": 100.0,
            "combined_score": 0.0,
        }

    else:
        gt_map = {item["id"]: item for item in ground_truth}

        # --- 2. Calculate F1 and Accuracy ---
        f1_premises = calculate_f1_premises(gt_map, predictions)

        common_args = {
            "ground_truth_list": ground_truth,
            "predictions_list": predictions,
            "metric_name": "validity",
            "prediction_key": "validity",
        }
        overall_acc, _, _ = calculate_accuracy(**common_args, plausibility_filter=None)

        # --- 3. Calculate Conditional Accuracies for Bias Metrics ---
        acc_plausible_valid, _, _ = calculate_subgroup_accuracy(
            gt_map, predictions, gt_validity=True, gt_plausibility=True
        )
        acc_implausible_valid, _, _ = calculate_subgroup_accuracy(
            gt_map, predictions, gt_validity=True, gt_plausibility=False
        )
        acc_plausible_invalid, _, _ = calculate_subgroup_accuracy(
            gt_map, predictions, gt_validity=False, gt_plausibility=True
        )
        acc_implausible_invalid, _, _ = calculate_subgroup_accuracy(
            gt_map, predictions, gt_validity=False, gt_plausibility=False
        )

        conditional_accuracies = {
            "acc_plausible_valid": acc_plausible_valid,
            "acc_implausible_valid": acc_implausible_valid,
            "acc_plausible_invalid": acc_plausible_invalid,
            "acc_implausible_invalid": acc_implausible_invalid,
        }

        # --- 4. Calculate Content Effect Bias Metrics ---
        bias_metrics = calculate_content_effect_bias(conditional_accuracies)
        tot_content_effect = bias_metrics["tot_content_effect"]

        # --- 5. Calculate Combined Metric ---
        # Only factor in F1 when ground truth has premise annotations
        has_premises = any("relevant_premises" in item for item in ground_truth)
        if has_premises:
            overall_performance_metric = (overall_acc + f1_premises) / 2.0
        else:
            overall_performance_metric = overall_acc
        smoother_score = calculate_smooth_combined_metric(
            overall_performance_metric, tot_content_effect
        )

        # --- 6. Prepare Final Output Dictionary ---
        final_results = {
            "accuracy": round(overall_acc, 4),
            "f1_premises": round(f1_premises, 4),
            "content_effect": round(tot_content_effect, 4),
            "combined_score": round(smoother_score, 4),
        }

    # --- 7. Write Results ---
    try:
        with open(output_path, "w") as f:
            json.dump(final_results, f)
            if verbose:
                print(f"Scoring complete. Results written to {output_path}")
    except Exception as e:
        print(f"Error writing final results to file: {e}", file=sys.stderr)
