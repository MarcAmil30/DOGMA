import json
import csv
from pathlib import Path

from proto_tools.tools.sequence_scoring.alphagenome.shared_data_models import (
    AlphaGenomeVariant,
)
from proto_tools.tools.sequence_scoring.alphagenome.alphagenome_predict_variants import (
    run_alphagenome_predict_variants,
    AlphaGenomePredictVariantsInput,
    AlphaGenomePredictVariantsConfig,
)
from proto_tools.tools.sequence_scoring.alphagenome.alphagenome_score_variants import (
    run_alphagenome_score_variants,
    AlphaGenomeScoreVariantsInput,
    AlphaGenomeScoreVariantsConfig,
)


# ============================================================
# Toy AlphaGenome variant inside Proto
# ============================================================
# This is a minimal test example.
# It is NOT biologically meaningful because chr1:0-16384 is just a toy interval.
#
# AlphaGenome/Proto variant tools need:
#   chromosome
#   interval_start
#   interval_end
#   variant_position
#   reference_bases
#   alternate_bases
#
# The variant must sit inside the interval.
# Smallest supported context length is 16,384 bp.
# ============================================================

variant = AlphaGenomeVariant(
    chromosome="chr1",
    interval_start=0,
    interval_end=16_384,
    variant_position=8_192,
    reference_bases="A",
    alternate_bases="G",
)


# ============================================================
# 1. Predict variant effects
# ============================================================

print("\nRunning AlphaGenome variant prediction through Proto...")

predict_result = run_alphagenome_predict_variants(
    AlphaGenomePredictVariantsInput(
        variants=[variant]
    ),
    AlphaGenomePredictVariantsConfig(
        requested_outputs=[
            "RNA_SEQ",
            "CAGE",
            "ATAC",
            "SPLICE_SITES",
        ],
        organism="human",
        device="cuda",
        verbose=1,
        timeout=3600,
    ),
)

print("\nPrediction finished.")
print(f"Number of prediction results: {len(predict_result.results)}")

pred = predict_result.results[0]

print("\nPrediction metadata:")
print(f"Chromosome: {pred.chromosome}")
print(f"Interval: {pred.interval_start}-{pred.interval_end}")
print(f"Variant: {pred.variant}")
print(f"Requested outputs: {pred.requested_outputs}")


# Save full prediction result as JSON
prediction_json = Path("alphagenome_proto_variant_prediction.json")
with open(prediction_json, "w") as f:
    json.dump(predict_result.model_dump(mode="json"), f, indent=2)

print(f"\nSaved full prediction output to: {prediction_json}")


# ============================================================
# 2. Score variant effects
# ============================================================

print("\nRunning AlphaGenome variant scoring through Proto...")

score_result = run_alphagenome_score_variants(
    AlphaGenomeScoreVariantsInput(
        variants=[variant]
    ),
    AlphaGenomeScoreVariantsConfig(
        # None = use all recommended AlphaGenome variant scorers
        variant_scorers=None,
        organism="human",
        device="cuda",
        verbose=1,
        timeout=3600,
    ),
)

print("\nScoring finished.")
print(f"Number of scoring results: {len(score_result.results)}")

scores = score_result.results[0].scores

print(f"Number of score rows: {len(scores)}")

# Show first few score records
print("\nFirst 5 score rows:")
for row in scores[:5]:
    print(row)


# Save scores as JSON
scores_json = Path("alphagenome_proto_variant_scores.json")
with open(scores_json, "w") as f:
    json.dump(score_result.model_dump(mode="json"), f, indent=2)

print(f"\nSaved full score output to: {scores_json}")


# Save scores as CSV
scores_csv = Path("alphagenome_proto_variant_scores.csv")

if scores:
    fieldnames = sorted(scores[0].keys())
    with open(scores_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scores)

    print(f"Saved score table to: {scores_csv}")
else:
    print("No scores returned.")
