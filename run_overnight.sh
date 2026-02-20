#!/bin/bash
# Overnight analysis suite — Zimmerman 14-tool + Sobol sensitivity
# Run with: bash run_overnight.sh
# Expected duration: ~20-30 minutes
# Output: artifacts/zimmerman/*.json, output/zimmerman/*.png

set -e

PROJECT="/Users/gardenofcomputation/how-to-live-much-longer"
cd "$PROJECT"

source /Users/gardenofcomputation/miniforge3/etc/profile.d/conda.sh
conda activate er

TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/overnight_${TIMESTAMP}.log"

echo "=== Overnight Analysis Suite ===" | tee "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "Project: $PROJECT" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Phase 1: Full Zimmerman 14-tool suite (n_base=256 for publication-quality Sobol)
echo "--- Phase 1: Zimmerman 14-tool suite (n_base=256) ---" | tee -a "$LOG_FILE"
START=$(date +%s)
python zimmerman_analysis.py --n-base 256 --viz 2>&1 | tee -a "$LOG_FILE"
END=$(date +%s)
echo "Phase 1 elapsed: $(($END - $START)) seconds" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Phase 2: Intervention-only Sobol for near_cliff_80 patient
echo "--- Phase 2: Sobol for near_cliff_80 (6D, n_base=256) ---" | tee -a "$LOG_FILE"
START=$(date +%s)
python zimmerman_analysis.py --tools sobol --patient near_cliff_80 --n-base 256 --viz 2>&1 | tee -a "$LOG_FILE"
END=$(date +%s)
echo "Phase 2 elapsed: $(($END - $START)) seconds" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Summary
echo "=== Overnight Analysis Complete ===" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Artifacts:" | tee -a "$LOG_FILE"
ls -lh artifacts/zimmerman/*.json 2>/dev/null | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Plots:" | tee -a "$LOG_FILE"
ls -lh output/zimmerman/*.png 2>/dev/null | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
