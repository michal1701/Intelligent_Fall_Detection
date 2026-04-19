#!/usr/bin/env bash
#cd SAFE
#TABLE3_ONE_BATCH=1 ./run_table3_paper_batches.sh
# SAFE Table 3 — paper sampling rates (48 / 32 / 16 / 8 / 4 kHz), one batch per process.
#
# Each run of main.py saves BEFORE the next batch starts:
#   - results_table3/ml_results_raw_audio_cv_<sr>hz.csv
#   - results_table3/table3_top10_<sr>hz.csv
#   - results_table3/ml_results_raw_audio_cv_all_rates.csv  (merged: keeps prior rates + this rate)
#   - results_table3/table3_checkpoint.log
#
# If the machine crashes, completed rates stay on disk. Re-run this script to continue.
#
# Usage (from SAFE/):
#   ./run_table3_paper_batches.sh
#
# Run only the next missing rate, then exit (safest for long jobs):
#   TABLE3_ONE_BATCH=1 ./run_table3_paper_batches.sh
#
# Redo a rate even if CSV already exists:
#   TABLE3_FORCE=1 ./run_table3_paper_batches.sh
#
# Extra args are forwarded to main.py, e.g.:
#   ./run_table3_paper_batches.sh --table3-no-gradient-boosting
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RATES=(48000 32000 16000 8000 4000)
OUT_DIR="results_table3"
MERGED_CSV="${OUT_DIR}/ml_results_raw_audio_cv_paper_merged.csv"

# Python resolution order:
# 1) explicit PYTHON env var
# 2) local project virtualenv (.venv/bin/python)
# 3) system python3 from PATH
if [[ -n "${PYTHON:-}" ]]; then
  :
elif [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${SCRIPT_DIR}/.venv/bin/python"
else
  PYTHON="$(command -v python3 || true)"
fi

if [[ -z "${PYTHON}" ]]; then
  echo "Error: python3 not found in PATH. Install Python 3 and retry." >&2
  exit 1
fi
echo "Using Python interpreter: ${PYTHON}"
"${PYTHON}" --version || true
SKIP_DONE="${TABLE3_SKIP_DONE:-1}"
FORCE="${TABLE3_FORCE:-0}"
ONE_BATCH="${TABLE3_ONE_BATCH:-0}"

run_one_rate() {
  local sr="$1"
  shift
  echo ""
  echo "================================================================================"
  echo " Table 3 batch: ${sr} Hz  |  $(date -Iseconds 2>/dev/null || date)"
  echo "================================================================================"
  "$PYTHON" main.py \
    --table 3 \
    --table3-sample-rates "${sr}" \
    --cv-folds 10 \
    --raw-duration-sec 3.0 \
    --data-dir "../data" \
    "$@"
  echo ""
  echo ">>> Batch ${sr} Hz finished — outputs are on disk. Safe to stop here before the next rate."
}

rate_output_path() {
  local sr="$1"
  echo "${OUT_DIR}/ml_results_raw_audio_cv_${sr}hz.csv"
}

should_skip() {
  local sr="$1"
  [[ "${FORCE}" -eq 1 ]] && return 1
  local f
  f="$(rate_output_path "${sr}")"
  [[ -f "${f}" ]] && [[ -s "${f}" ]]
}

first_missing_rate() {
  local sr
  for sr in "${RATES[@]}"; do
    if ! should_skip "${sr}"; then
      echo "${sr}"
      return 0
    fi
  done
  return 1
}

merge_csvs() {
  echo ""
  echo "================================================================================"
  echo " Optional merge (header + concat) -> ${MERGED_CSV}"
  echo " (You already have ml_results_raw_audio_cv_all_rates.csv from main.py)"
  echo "================================================================================"
  mkdir -p "${OUT_DIR}"
  local first=1
  local missing=0
  for sr in "${RATES[@]}"; do
    local f="${OUT_DIR}/ml_results_raw_audio_cv_${sr}hz.csv"
    if [[ ! -f "${f}" ]]; then
      echo "Warning: missing ${f} (skip in merge)" >&2
      missing=1
      continue
    fi
    if [[ "${first}" -eq 1 ]]; then
      cp "${f}" "${MERGED_CSV}"
      first=0
    else
      tail -n +2 "${f}" >> "${MERGED_CSV}"
    fi
  done
  if [[ "${first}" -eq 1 ]]; then
    echo "No per-rate CSVs to merge." >&2
    return 0
  fi
  echo "Wrote: ${MERGED_CSV}"
  if [[ "${missing}" -ne 0 ]]; then
    echo "Note: merge incomplete (some rates missing)." >&2
  fi
}

# --- main ---
mkdir -p "${OUT_DIR}"

if [[ "${ONE_BATCH}" -eq 1 ]]; then
  SR="$(first_missing_rate || true)"
  if [[ -z "${SR}" ]]; then
    echo "All paper rates already have saved CSVs under ${OUT_DIR}/"
    echo "Set TABLE3_FORCE=1 to recompute a rate."
    merge_csvs
    exit 0
  fi
  echo "ONE-BATCH mode: running only ${SR} Hz (then exiting)."
  run_one_rate "${SR}" "$@"
  merge_csvs
  echo ""
  echo "Done. Run again:  TABLE3_ONE_BATCH=1 ./run_table3_paper_batches.sh"
  exit 0
fi

for sr in "${RATES[@]}"; do
  if should_skip "${sr}"; then
    echo "Skip ${sr} Hz (already saved). Set TABLE3_FORCE=1 to redo."
    continue
  fi
  run_one_rate "${sr}" "$@"
done

merge_csvs
echo ""
echo "All Table 3 paper batches finished (or skipped if already present)."
