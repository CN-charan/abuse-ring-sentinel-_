"""
Layer 4: Evaluation & Cost Reporting
--------------------------------------
Turns raw prediction counts into the metrics judges actually asked for:
precision, recall, F1, and false-positive/false-negative COST -- not
just accuracy, which is misleading here since only 5% of accounts
are actually fraud (class imbalance).

Cost assumptions (illustrative, stated explicitly -- not researched
real-world figures):
  False positive (innocent account wrongly flagged) -> $10
      (customer friction: support ticket, account review, trust hit)
  False negative (ring member missed)               -> $100
      (direct fraud loss that goes undetected)
"""

import pandas as pd

COST_PER_FALSE_POSITIVE = 10
COST_PER_FALSE_NEGATIVE = 100


def main():
    predictions = pd.read_csv("predictions.csv")
    ground_truth = pd.read_csv("ground_truth.csv")
    merged = predictions.merge(ground_truth, on="account_id")

    tp = ((merged.predicted_flag == 1) & (merged.is_fraud_ring == 1)).sum()
    fp = ((merged.predicted_flag == 1) & (merged.is_fraud_ring == 0)).sum()
    fn = ((merged.predicted_flag == 0) & (merged.is_fraud_ring == 1)).sum()
    tn = ((merged.predicted_flag == 0) & (merged.is_fraud_ring == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    accuracy = (tp + tn) / len(merged)

    fp_cost = fp * COST_PER_FALSE_POSITIVE
    fn_cost = fn * COST_PER_FALSE_NEGATIVE
    total_cost = fp_cost + fn_cost

    print("=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)
    print(f"  True Positives:  {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Negatives:  {tn}")

    print("\n" + "=" * 50)
    print("METRICS")
    print("=" * 50)
    print(f"  Accuracy:  {accuracy:.3f}   <- misleading alone, 95% are non-fraud")
    print(f"  Precision: {precision:.3f}  (of flagged accounts, how many were real rings)")
    print(f"  Recall:    {recall:.3f}  (of real ring members, how many we caught)")
    print(f"  F1 Score:  {f1:.3f}")

    print("\n" + "=" * 50)
    print("FALSE-POSITIVE / FALSE-NEGATIVE COST")
    print("=" * 50)
    print(f"  FP cost: {fp} x ${COST_PER_FALSE_POSITIVE} = ${fp_cost}")
    print(f"  FN cost: {fn} x ${COST_PER_FALSE_NEGATIVE} = ${fn_cost}")
    print(f"  TOTAL ESTIMATED COST: ${total_cost}")
    print("\n  (Cost figures are illustrative assumptions, not researched")
    print("   real-world data -- stated explicitly for transparency.)")

    # Save a small summary file too, useful for the README/pitch
    summary = pd.DataFrame([{
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
        "fp_cost": fp_cost, "fn_cost": fn_cost, "total_cost": total_cost,
    }])
    summary.to_csv("evaluation_summary.csv", index=False)
    print("\nSaved evaluation_summary.csv")


if __name__ == "__main__":
    main()
