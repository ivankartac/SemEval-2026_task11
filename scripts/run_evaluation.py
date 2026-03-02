import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.evaluation import run_full_scoring
from tqdm import tqdm


def prepare_and_run(targets_path, predictions_path, final_output, n_iterations=1000):
    # 1. Load and align data
    with open(predictions_path, "r") as f:
        preds = [json.loads(line) for line in f if line.strip()]
        for p in preds:
            p["validity"] = p.get("valid", p.get("validity"))

    with open(targets_path, "r") as f:
        full_gt = json.load(f)

    pred_map = {str(p["id"]): p for p in preds}
    pairs = [(pred_map[str(g["id"])], g) for g in full_gt if str(g["id"]) in pred_map]
    n = len(pairs)

    # 2. Bootstrap iteration
    history = {"accuracy": [], "f1_premises": [], "content_effect": [], "combined_score": []}
    tmp_p, tmp_t, tmp_o = "tmp_p.json", "tmp_t.json", "tmp_o.json"

    for _ in tqdm(range(n_iterations), desc="Bootstrapping"):
        sample = [pairs[i] for i in np.random.randint(0, n, n)]

        with open(tmp_p, "w") as f:
            json.dump([s[0] for s in sample], f)
        with open(tmp_t, "w") as f:
            json.dump([s[1] for s in sample], f)

        run_full_scoring(tmp_t, tmp_p, tmp_o, verbose=False)

        with open(tmp_o, "r") as f:
            res = json.load(f)
            for key in history:
                history[key].append(res.get(key, 0))

    # 3. Stats & output
    stats = {}
    print("\n" + "=" * 55)
    print(f"{'METRIC':<18} | {'MEAN':<8} | {'STD':<6} | {'95% CI'}")
    print("-" * 55)

    for k, v in history.items():
        m, s = np.mean(v), np.std(v)
        ci = [np.percentile(v, 2.5), np.percentile(v, 97.5)]
        stats[k] = {
            "mean": round(m, 3),
            "std": round(s, 3),
            "ci": [round(c, 3) for c in ci],
        }
        print(f"{k.upper():<18} | {m:<8.2f} | {s:<6.2f} | [{ci[0]:.2f}, {ci[1]:.2f}]")

    with open(final_output, "w") as f:
        json.dump(stats, f, indent=4)
    for f in [tmp_p, tmp_t, tmp_o]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions")
    parser.add_argument("targets")
    parser.add_argument("-o", "--output")
    parser.add_argument("-i", "--iterations", type=int, default=1000)
    args = parser.parse_args()
    prepare_and_run(args.targets, args.predictions, args.output, args.iterations)
