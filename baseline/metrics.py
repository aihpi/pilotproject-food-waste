"""Compute returned-weight metrics for every llm_inference_results*.json in this folder."""
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def returned(item):
    # Accept 'returned_weight', 'Returned Weight', and the older 'returned_amount' key.
    for k, v in item.items():
        if k.lower().replace(" ", "_") in ("returned_weight", "returned_amount"):
            try:
                return float(str(v).lower().replace("g", "").strip())
            except ValueError:
                return None
    return None


def per_ingredient(data):
    if not isinstance(data, dict) or not isinstance(data.get("ingredients"), list):
        return None
    out = {}
    for item in data["ingredients"]:
        if isinstance(item, dict) and (w := returned(item)) is not None:
            out[str(item.get("name", "")).strip().lower()] = w
    return out


def metrics(results):
    tot_err, ing_err, gt_nonzero, failed = [], [], [], 0
    for r in results:
        pred, gt = per_ingredient(r.get("prediction")), per_ingredient(r.get("ground_truth"))
        if pred is None or gt is None:
            failed += 1
            continue
        p, a = sum(pred.values()), sum(gt.values())
        tot_err.append(p - a)
        if a:
            gt_nonzero.append(abs(p - a) / a)
        # Ingredient missing from the prediction counts as a predicted 0g.
        ing_err += [pred.get(name, 0.0) - w for name, w in gt.items()]
    n = len(tot_err)
    mean = lambda xs: sum(xs) / len(xs) if xs else math.nan
    return {
        "samples": len(results),
        "parsed": n,
        "failed": failed,
        "MAE (g)": mean([abs(e) for e in tot_err]),
        "RMSE (g)": math.sqrt(mean([e * e for e in tot_err])),
        "MAPE (%)": mean(gt_nonzero) * 100,
        "Bias (g)": mean(tot_err),
        "Ingredient MAE (g)": mean([abs(e) for e in ing_err]),
    }


if __name__ == "__main__":
    for f in sorted(glob.glob(os.path.join(HERE, "llm_inference_results*.json"))):
        with open(f, encoding="utf-8") as fh:
            m = metrics(json.load(fh))
        print(os.path.basename(f))
        for k, v in m.items():
            print(f"  {k:<20} {v:.2f}" if isinstance(v, float) else f"  {k:<20} {v}")
