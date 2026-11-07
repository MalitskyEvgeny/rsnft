#!/usr/bin/env bash
# Прямая задача + HTML-отчёт (report_full.jl).
#
# Основной режим — АДАПТИВНАЯ СЕТКА (adaptive_basin_fvm.jl). Флаг --uniform
# переключает на классический равномерный солвер (basin_fvm.jl).
#
# Usage:
#   ./run_forward.sh                                        # адаптивная сетка по умолчанию
#   ./run_forward.sh --adaptive-base 30,60 --adaptive-level 1
#   ./run_forward.sh --uniform --nx 30 --nz 60 --force
#   ./run_forward.sh --data-dir ../Данные_для_прямой_задачи/data2 --open
#
# Флаги:
#   --data-dir DIR       данные прямой задачи (<repo>/Данные_для_прямой_задачи/data1)
#   --adaptive-base N,M  базовая сетка адаптивного солвера (16,32)
#   --adaptive-level N   максимальный уровень измельчения (2)
#   --uniform            равномерный солвер вместо адаптивного
#   --nx N --nz N        сетка равномерного солвера (60 120; только с --uniform)
#   --max-dt-ma F        максимальный шаг по времени, Ма (2.5)
#   --picard-iters N     итерации Пикара (3)
#   --threads N          Julia-потоки (8)
#   --out FILE           куда писать отчёт (results/report_forward_<timestamp>/report.html)
#   --force              пересчитать, даже если результат в кэше
#   --open               открыть отчёт в браузере
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R3="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$R3/.." && pwd)"
JULIA=${JULIA:-julia}
# Флаги производительности: максимум оптимизации и отказ от проверок границ
# (индексы везде @inbounds). Отключается переменной RSNFT_SAFE=1.
if [[ "${RSNFT_SAFE:-0}" != "1" ]]; then
    JULIA_PERF_FLAGS="-O3 --check-bounds=no"
else
    JULIA_PERF_FLAGS=""
fi
THREADS=8

DATA_DIR=""
ADAPTIVE_BASE="16,32"
ADAPTIVE_LEVEL=2
UNIFORM=0
NX=60
NZ=120
MAX_DT_MA=2.5
PICARD=3
OUT=""
FORCE=0
OPEN=0

# Git Bash передаёт native-программам POSIX-пути; конвертируем для Julia.
to_julia_path() {
    command -v cygpath >/dev/null 2>&1 && cygpath -m "$1" || printf '%s' "$1"
}

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)       DATA_DIR="$2"; shift 2 ;;
        --adaptive-base)  ADAPTIVE_BASE="$2"; shift 2 ;;
        --adaptive-level) ADAPTIVE_LEVEL="$2"; shift 2 ;;
        --uniform)        UNIFORM=1; shift ;;
        --nx)             NX="$2"; shift 2 ;;
        --nz)             NZ="$2"; shift 2 ;;
        --max-dt-ma)      MAX_DT_MA="$2"; shift 2 ;;
        --picard-iters)   PICARD="$2"; shift 2 ;;
        --threads)        THREADS="$2"; shift 2 ;;
        --out)            OUT="$2"; shift 2 ;;
        --force)          FORCE=1; shift ;;
        --open)           OPEN=1; shift ;;
        -h|--help)        usage ;;
        *) echo "unknown flag: $1" >&2; usage ;;
    esac
done

[[ -n "$DATA_DIR" ]] || DATA_DIR="$REPO_ROOT/Данные_для_прямой_задачи/data1"
[[ -d "$DATA_DIR" ]] || { echo "data dir not found: $DATA_DIR" >&2; exit 1; }
DATA_DIR="$(to_julia_path "$DATA_DIR")"
CACHE_DIR="$(to_julia_path "$REPO_ROOT/results")"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT:-$REPO_ROOT/results/report_forward_$TS}"
mkdir -p "$OUT_DIR"
OUT_DIR="$(to_julia_path "$OUT_DIR")"

SOLVER="$R3/adaptive_basin_fvm.jl"
if [[ $UNIFORM -eq 1 ]]; then
    SOLVER="$R3/basin_fvm.jl"
    echo "=== Stage 1/2: forward solve (uniform): $(basename "$DATA_DIR"), grid ${NX}x${NZ} ==="
    SOLVER_ARGS=(--data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR"
                 --nx "$NX" --nz "$NZ" --max-dt-ma "$MAX_DT_MA"
                 --picard-iterations "$PICARD")
else
    echo "=== Stage 1/2: forward solve (adaptive): $(basename "$DATA_DIR"), base $ADAPTIVE_BASE, level $ADAPTIVE_LEVEL ==="
    SOLVER_ARGS=(--data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR"
                 --base-nx "${ADAPTIVE_BASE%,*}" --base-nz "${ADAPTIVE_BASE#*,}"
                 --max-level "$ADAPTIVE_LEVEL" --max-dt-ma "$MAX_DT_MA"
                 --picard-iterations "$PICARD")
fi
[[ $FORCE -eq 1 ]] && SOLVER_ARGS+=(--force)

RUN_LINE="$("$JULIA" --project="$R3" $JULIA_PERF_FLAGS -t "$THREADS" "$SOLVER" "${SOLVER_ARGS[@]}" | tee /dev/stderr | grep -E '^(RUN|CACHE_HIT)' | tail -1)"
RUN_ID="$(printf '%s' "$RUN_LINE" | tr -d '[:cntrl:]' | awk '{print $2}' | tr -d ':')"
RUN_DIR="$(to_julia_path "$REPO_ROOT/results/$RUN_ID")"
echo "run dir: $RUN_DIR"

echo "=== Stage 2/2: report ==="
"$JULIA" --project="$R3" $JULIA_PERF_FLAGS "$R3/report_full.jl" --direct "$RUN_DIR" --out "$OUT_DIR/report.html"

echo "=== summary ==="
echo "report   : $OUT_DIR/report.html"
echo "run dir  : $RUN_DIR"
[[ $OPEN -eq 1 ]] && (xdg-open "$OUT_DIR/report.html" >/dev/null 2>&1 ||
                      open "$OUT_DIR/report.html" >/dev/null 2>&1 ||
                      echo "open the report manually: $OUT_DIR/report.html")
exit 0
