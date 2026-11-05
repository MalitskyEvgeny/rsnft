#!/usr/bin/env bash
# Adaptive-grid inverse run with report, mesh pictures and a final summary.
# Bash counterpart of run_adaptive.ps1: wraps run_inverse.jl (adaptive backend)
# + report_full.jl (inverse section), then prints acceptance numbers from solution.json.
#
# Usage:
#   ./run_adaptive.sh                              # final mode, defaults
#   ./run_adaptive.sh --mode search --adaptive-base 16,32 --adaptive-level 1
#   ./run_adaptive.sh --grid 30,60 --adaptive-base 30,60 --adaptive-level 1
set -euo pipefail

REPO_ROOT=""
JULIA=julia
DATA_DIR=""
WELLS=""
MODE=final
OUT_DIR=""
GRID="20,40"
ADAPTIVE_BASE="16,32"
ADAPTIVE_LEVEL=2
ADAPTIVE_PREVIEW="120,160"
THETA0_FILE=""
THREADS=8
ENSEMBLE=1
WORKERS=1
DE_BUDGET=2000
NM_ITERATIONS=30
REG=1.0
SEED=1
SIGMA_T=2.0
SIGMA_RO=0.05
MAX_DT_MA=2.5
PCG_TOL=1e-4
FROZEN=0
NO_REPORT=0

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-root)        REPO_ROOT="$2"; shift 2 ;;
        --julia)            JULIA="$2"; shift 2 ;;
        --data-dir)         DATA_DIR="$2"; shift 2 ;;
        --wells)            WELLS="$2"; shift 2 ;;
        --mode)             MODE="$2"; shift 2 ;;
        --out-dir)          OUT_DIR="$2"; shift 2 ;;
        --grid)             GRID="$2"; shift 2 ;;
        --adaptive-base)    ADAPTIVE_BASE="$2"; shift 2 ;;
        --adaptive-level)   ADAPTIVE_LEVEL="$2"; shift 2 ;;
        --adaptive-preview) ADAPTIVE_PREVIEW="$2"; shift 2 ;;
        --theta0-file)      THETA0_FILE="$2"; shift 2 ;;
        --threads)          THREADS="$2"; shift 2 ;;
        --ensemble)         ENSEMBLE="$2"; shift 2 ;;
        --workers)          WORKERS="$2"; shift 2 ;;
        --de-budget)        DE_BUDGET="$2"; shift 2 ;;
        --nm-iterations)    NM_ITERATIONS="$2"; shift 2 ;;
        --reg)              REG="$2"; shift 2 ;;
        --seed)             SEED="$2"; shift 2 ;;
        --sigma-T)          SIGMA_T="$2"; shift 2 ;;
        --sigma-Ro)         SIGMA_RO="$2"; shift 2 ;;
        --max-dt-ma)        MAX_DT_MA="$2"; shift 2 ;;
        --pcg-tol)          PCG_TOL="$2"; shift 2 ;;
        --frozen-mesh)      FROZEN=1; shift ;;
        --no-report)        NO_REPORT=1; shift ;;
        -h|--help)          usage ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 1 ;;
    esac
done

case "$MODE" in
    final|search) ;;
    *) echo "--mode must be final or search" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/ lives at <repo>/rsnft3/scripts, so the repo root is two levels up.
if [[ -z "$REPO_ROOT" ]]; then REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"; fi
R3="$REPO_ROOT/rsnft3"

command -v "$JULIA" >/dev/null 2>&1 || { echo "julia not found on PATH" >&2; exit 1; }
[[ -f "$R3/Project.toml" ]] || { echo "rsnft3 project not found under $REPO_ROOT" >&2; exit 1; }
if [[ -z "$DATA_DIR" ]]; then DATA_DIR="$REPO_ROOT/Данные для обратной задачи"; fi
[[ -d "$DATA_DIR" ]] || { echo "data dir not found: $DATA_DIR" >&2; exit 1; }
if [[ -n "$THETA0_FILE" && ! -f "$THETA0_FILE" ]]; then echo "theta0-file not found: $THETA0_FILE" >&2; exit 1; fi

if [[ -z "$WELLS" ]]; then
    WELLS="$DATA_DIR/TemperatureWellData.xlsx;$DATA_DIR/VitriniteWellData.xlsx"
fi
if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$REPO_ROOT/results/adaptive_${MODE}_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$OUT_DIR"

ADAPTIVE_FLAG="--adaptive-final"
[[ "$MODE" == "search" ]] && ADAPTIVE_FLAG="--adaptive-search"
FROZEN_FLAG=()
[[ "$FROZEN" == 1 ]] && FROZEN_FLAG=(--adaptive-frozen)
THETA0_FLAG=()
[[ -n "$THETA0_FILE" ]] && THETA0_FLAG=(--theta0-file "$(cd "$(dirname "$THETA0_FILE")" && pwd)/$(basename "$THETA0_FILE")")

echo "=== run_inverse ($MODE) ==="
echo "out=$OUT_DIR  grid=$GRID  adaptive base=$ADAPTIVE_BASE level=$ADAPTIVE_LEVEL preview=$ADAPTIVE_PREVIEW"
T0=$SECONDS

julia --project="$R3" -t "$THREADS" "$R3/run_inverse.jl" \
    --data-dir "$DATA_DIR" \
    --wells "$WELLS" \
    --grid "$GRID" \
    --ensemble "$ENSEMBLE" --workers "$WORKERS" \
    --de-budget "$DE_BUDGET" --nm-iterations "$NM_ITERATIONS" \
    --sigma-T "$SIGMA_T" --sigma-Ro "$SIGMA_RO" \
    --reg "$REG" --seed "$SEED" \
    --max-dt-ma "$MAX_DT_MA" --pcg-tol "$PCG_TOL" \
    --adaptive-base "$ADAPTIVE_BASE" --adaptive-level "$ADAPTIVE_LEVEL" \
    --adaptive-preview "$ADAPTIVE_PREVIEW" \
    "$ADAPTIVE_FLAG" \
    ${FROZEN_FLAG+"${FROZEN_FLAG[@]}"} \
    ${THETA0_FLAG+"${THETA0_FLAG[@]}"} \
    --out "$OUT_DIR"

echo "inverse done in $(( SECONDS - T0 ))s"
SOLUTION="$OUT_DIR/solution.json"
[[ -f "$SOLUTION" ]] || { echo "solution.json missing: $SOLUTION" >&2; exit 1; }

if [[ "$NO_REPORT" == 0 ]]; then
    echo ""
    echo "=== report ==="
    julia --project="$R3" "$R3/report_full.jl" --inverse "$OUT_DIR" \
        || echo "report_full.jl failed; continuing" >&2
fi

echo ""
echo "=== summary ==="
"$JULIA" --project="$R3" -e '
const sol = JSON.parsefile(ARGS[1])
b, cr = sol["best"], sol["criteria"]
th = join([round(v, digits=2) for v in b["theta"]], ", ")
p2(x) = string(round(100x, digits=1), "%")
r2(x) = string(round(x, digits=2))
println("status   : ", sol["status"])
println("best J   : ", round(b["J"], digits=1), "  (J_T=", round(b["J_T"], digits=1),
        ", J_Ro=", round(b["J_Ro"], digits=2), ", J_reg=", round(b["J_reg"], digits=1), ")")
println("rmse_T   : ", r2(cr["rmse_T"]), "   (accept: <= 4 = 2*sigma_T)")
println("rmse_Ro  : ", r2(cr["rmse_Ro"]), "  (accept: <= 0.10)")
println("T in 2s  : ", p2(cr["frac_2s_T"]), "  (accept: >= 95%)")
println("Ro in 2s : ", p2(cr["frac_2s_Ro"]))
if haskey(sol, "adaptive")
    a = sol["adaptive"]["criteria"]
    println("adaptive : base=", sol["adaptive"]["base"], " level=", sol["adaptive"]["level"],
            "  rmse_T=", r2(a["rmse_T"]), "  rmse_Ro=", r2(a["rmse_Ro"]),
            "  T in 2s=", p2(a["frac_2s_T"]))
end
println("theta    : ", th)
' "$SOLUTION"

[[ -f "$OUT_DIR/mesh_adaptive.png" ]] && echo "mesh     : $OUT_DIR/mesh_adaptive.png"
[[ -f "$OUT_DIR/report.html"     ]] && echo "report   : $OUT_DIR/report.html"
echo "total    : $(( SECONDS - T0 ))s"
