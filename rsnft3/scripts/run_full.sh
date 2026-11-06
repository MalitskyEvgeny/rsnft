#!/usr/bin/env bash
# Прямая + обратная задача + единый HTML-отчёт (report_full.jl).
# Прямой прогон даёт секцию «Прямая задача», инверсия — «Обратная задача».
#
# Основной режим — АДАПТИВНАЯ СЕТКА:
#   * прямой прогон — adaptive_basin_fvm.jl (--base/--level);
#   * инверсия — контроль на адаптивной сетке (run_inverse.jl --adaptive-final).
# Флаг --uniform переключает всё на классические равномерные сетки.
#
# Usage:
#   ./run_full.sh --grid 30,60 --de-budget 200          # быстрый контрольный прогон
#   ./run_full.sh --twin --q 75,80,85,90                # twin-режим
#   ./run_full.sh --data-dir ... --wells a.xlsx;b.xlsx  # реальные данные
#
# Основные флаги:
#   --data-dir DIR      данные прямой задачи (<repo>/Данные_для_прямой_задачи/data1)
#   --inv-data-dir DIR  данные для инверсии (<repo>/"Данные для обратной задачи")
#   --wells LIST        скважины для инверсии, через ";"
#                       (по умолчанию Temperature+Vitrinite из --inv-data-dir)
#   --twin              сгенерировать twin-наблюдения перед инверсией (make_twin.jl)
#   --q LIST            истинный q для twin, по значению на эпоху (75,80,85,90)
#   --base N,M          адаптивная база прямой задачи (16,32)
#   --level N           адаптивный уровень прямой задачи (2)
#   --adaptive-base N,M адаптивная база инверсии (16,32)
#   --adaptive-level N  адаптивный уровень инверсии (2)
#   --uniform           равномерные сетки вместо адаптивных
#   --nx N --nz N       сетка равномерной прямой задачи (60 120; только с --uniform)
#   --ensemble N (3) --workers N --de-budget N (4000) --nm-iterations N (200) --threads N
#   --sigma-T F --sigma-Ro F --reg F --seed N
#   --theta0-file FILE  полировка из готового solution.json (DE пропускается)
#   --out DIR           каталог отчёта (results/report_full_<timestamp>)
#   --force             пересчитать прямой прогон, даже если есть в кэше
#   --open              открыть отчёт в браузере
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R3="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$R3/.." && pwd)"
JULIA=${JULIA:-julia}

DATA_DIR=""
INV_DATA_DIR=""
WELLS=""
TWIN=0
TWIN_Q="75,80,85,90"
TWIN_SEED=42
GRID="20,40"
BASE="16,32"
LEVEL=2
ADAPTIVE_BASE="16,32"
ADAPTIVE_LEVEL=2
UNIFORM=0
NX=60
NZ=120
ENSEMBLE=3
WORKERS=1
DE_BUDGET=4000
THREADS=8
NM_ITERATIONS=200
SIGMA_T=2.0
SIGMA_RO=0.05
REG=1.0
SEED=1
THETA0=""
OUT=""
FORCE=0
OPEN=0

# Git Bash передаёт native-программам POSIX-пути; конвертируем для Julia.
to_julia_path() {
    command -v cygpath >/dev/null 2>&1 && cygpath -m "$1" || printf '%s' "$1"
}

to_julia_pathlist() {
    local out="" p
    local IFS=';'
    for p in $1; do
        out+="$(to_julia_path "$p");"
    done
    printf '%s' "${out%;}"
}

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)      DATA_DIR="$2"; shift 2 ;;
        --inv-data-dir)  INV_DATA_DIR="$2"; shift 2 ;;
        --wells)         WELLS="$2"; shift 2 ;;
        --twin)          TWIN=1; shift ;;
        --q)             TWIN_Q="$2"; shift 2 ;;
        --twin-seed)     TWIN_SEED="$2"; shift 2 ;;
        --grid)          GRID="$2"; shift 2 ;;
        --base)          BASE="$2"; shift 2 ;;
        --level)         LEVEL="$2"; shift 2 ;;
        --adaptive-base)  ADAPTIVE_BASE="$2"; shift 2 ;;
        --adaptive-level) ADAPTIVE_LEVEL="$2"; shift 2 ;;
        --uniform)       UNIFORM=1; shift ;;
        --nx)            NX="$2"; shift 2 ;;
        --nz)            NZ="$2"; shift 2 ;;
        --ensemble)      ENSEMBLE="$2"; shift 2 ;;
        --workers)       WORKERS="$2"; shift 2 ;;
        --de-budget)     DE_BUDGET="$2"; shift 2 ;;
        --threads)       THREADS="$2"; shift 2 ;;
        --nm-iterations) NM_ITERATIONS="$2"; shift 2 ;;
        --sigma-T)       SIGMA_T="$2"; shift 2 ;;
        --sigma-Ro)      SIGMA_RO="$2"; shift 2 ;;
        --reg)           REG="$2"; shift 2 ;;
        --seed)          SEED="$2"; shift 2 ;;
        --theta0-file)   THETA0="$2"; shift 2 ;;
        --out)           OUT="$2"; shift 2 ;;
        --force)         FORCE=1; shift ;;
        --open)          OPEN=1; shift ;;
        -h|--help)       usage ;;
        *) echo "unknown flag: $1" >&2; usage ;;
    esac
done

[[ -n "$DATA_DIR" ]] || DATA_DIR="$REPO_ROOT/Данные_для_прямой_задачи/data1"
[[ -n "$INV_DATA_DIR" ]] || INV_DATA_DIR="$REPO_ROOT/Данные для обратной задачи"
[[ -d "$DATA_DIR" ]] || { echo "data dir not found: $DATA_DIR" >&2; exit 1; }
[[ -d "$INV_DATA_DIR" ]] || { echo "inverse data dir not found: $INV_DATA_DIR" >&2; exit 1; }
[[ -n "$WELLS" ]] || WELLS="$INV_DATA_DIR/TemperatureWellData.xlsx;$INV_DATA_DIR/VitriniteWellData.xlsx"
DATA_DIR="$(to_julia_path "$DATA_DIR")"
INV_DATA_DIR="$(to_julia_path "$INV_DATA_DIR")"
WELLS="$(to_julia_pathlist "$WELLS")"
CACHE_DIR="$(to_julia_path "$REPO_ROOT/results")"
[[ -n "$THETA0" ]] && THETA0="$(to_julia_path "$THETA0")"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT:-$REPO_ROOT/results/report_full_$TS}"
mkdir -p "$OUT_DIR"
OUT_DIR="$(to_julia_path "$OUT_DIR")"

SOLVER="$R3/adaptive_basin_fvm.jl"
if [[ $UNIFORM -eq 1 ]]; then
    echo "=== Stage 1/3: forward solve (uniform): $(basename "$DATA_DIR"), grid ${NX}x${NZ} ==="
    SOLVER_ARGS=(--data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR"
                 --nx "$NX" --nz "$NZ")
    SOLVER="$R3/basin_fvm.jl"
else
    echo "=== Stage 1/3: forward solve (adaptive): $(basename "$DATA_DIR"), base $BASE, level $LEVEL ==="
    SOLVER_ARGS=(--data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR"
                 --base-nx "${BASE%,*}" --base-nz "${BASE#*,}"
                 --max-level "$LEVEL")
    SOLVER="$R3/adaptive_basin_fvm.jl"
fi
[[ $FORCE -eq 1 ]] && SOLVER_ARGS+=(--force)
RUN_LINE="$("$JULIA" --project="$R3" -t "$THREADS" "$SOLVER" "${SOLVER_ARGS[@]}" | tee /dev/stderr | grep -E '^(RUN|CACHE_HIT)' | tail -1)"
RUN_ID="$(printf '%s' "$RUN_LINE" | tr -d '[:cntrl:]' | awk '{print $2}' | tr -d ':')"
RUN_DIR="$(to_julia_path "$REPO_ROOT/results/$RUN_ID")"
echo "run dir: $RUN_DIR"

echo "=== Stage 2/3: inversion (wells: $WELLS) ==="
INVERSE_ARGS=(--data-dir "$INV_DATA_DIR" --wells "$WELLS" --grid "$GRID"
              --ensemble "$ENSEMBLE" --workers "$WORKERS"
              --de-budget "$DE_BUDGET" --nm-iterations "$NM_ITERATIONS"
              --sigma-T "$SIGMA_T" --sigma-Ro "$SIGMA_RO"
              --reg "$REG" --seed "$SEED"
              --out "$OUT_DIR/inverse")
if [[ $UNIFORM -eq 0 ]]; then
    INVERSE_ARGS+=(--adaptive-final --adaptive-base "$ADAPTIVE_BASE"
                   --adaptive-level "$ADAPTIVE_LEVEL")
fi
if [[ $TWIN -eq 1 ]]; then
    TWIN_FILE="$OUT_DIR/twin_grid.json"
    echo "--- twin: q=$TWIN_Q, grid $GRID ---"
    "$JULIA" --project="$R3" -t "$THREADS" "$R3/make_twin.jl" --data-dir "$INV_DATA_DIR" \
        --wells "${WELLS%%;*}" --grid "$GRID" --q "$TWIN_Q" \
        --sigma-T "$SIGMA_T" --sigma-Ro "$SIGMA_RO" --seed "$TWIN_SEED" \
        --out "$TWIN_FILE"
    INVERSE_ARGS=(--data-dir "$INV_DATA_DIR" --wells "$TWIN_FILE" "${INVERSE_ARGS[@]:2}")
fi
[[ $TWIN -eq 1 ]] && INVERSE_ARGS+=(--q-true "$TWIN_Q")
[[ -n "$THETA0" ]] && INVERSE_ARGS+=(--theta0-file "$THETA0")

"$JULIA" --project="$R3" -t "$THREADS" "$R3/run_inverse.jl" "${INVERSE_ARGS[@]}"

echo "=== Stage 3/3: report ==="
"$JULIA" --project="$R3" -t "$THREADS" "$R3/report_full.jl" \
    --direct "$RUN_DIR" --inverse "$OUT_DIR/inverse" --out "$OUT_DIR/report.html"

echo "=== summary ==="
echo "report   : $OUT_DIR/report.html"
echo "forward  : $RUN_DIR"
echo "inverse  : $OUT_DIR/inverse"
[[ $OPEN -eq 1 ]] && (xdg-open "$OUT_DIR/report.html" >/dev/null 2>&1 ||
                      open "$OUT_DIR/report.html" >/dev/null 2>&1 ||
                      echo "open the report manually: $OUT_DIR/report.html")
exit 0
