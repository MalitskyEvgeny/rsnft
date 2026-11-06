#!/usr/bin/env bash
# 6.4: дедуп и разбор results/ — отчёт об объёме, удаление ТОЛЬКО по явному
# подтверждению (--delete для каждой выбранной группы).
#
# Usage:
#   ./cleanup_results.sh                     # только отчёт об объёме (ничего не удаляет)
#   ./cleanup_results.sh --delete-dup-cache  # удалить дубли _input_cache внутри прогонов
#   ./cleanup_results.sh --delete-bench      # удалить временные benchmark-каталоги
#   ./cleanup_results.sh --delete-old N      # удалить прогоны старше N дней (кроме manifest)
set -euo pipefail

RESULTS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/results"
DELETE_DUP=0; DELETE_BENCH=0; DELETE_OLD=-1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --delete-dup-cache) DELETE_DUP=1; shift ;;
        --delete-bench)     DELETE_BENCH=1; shift ;;
        --delete-old)       DELETE_OLD="$2"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

[[ -d "$RESULTS" ]] || { echo "нет каталога $RESULTS" >&2; exit 0; }

echo "=== объём results/ ==="
du -sh "$RESULTS" 2>/dev/null || true

echo
echo "=== дубли _input_cache внутри прогонов (общий теперь results/_input_cache) ==="
find "$RESULTS" -mindepth 2 -maxdepth 2 -type d -name "_input_cache" -print | while read -r d; do
    du -sh "$d" || true
done
DUP_COUNT=$(find "$RESULTS" -mindepth 2 -maxdepth 2 -type d -name "_input_cache" | wc -l)
DUP_SIZE=$(du -shc "$RESULTS"/*/*/_input_cache 2>/dev/null | tail -1 | cut -f1)
echo "итого: $DUP_COUNT дублей, $DUP_SIZE"

if [[ $DELETE_DUP -eq 1 ]]; then
    find "$RESULTS" -mindepth 2 -maxdepth 2 -type d -name "_input_cache" -exec rm -rf {} +
    echo "удалено: $DUP_COUNT дублей _input_cache"
else
    echo "(удаление: --delete-dup-cache)"
fi

echo
echo "=== временные benchmark-каталоги (_bench_cache, _ageprobe, _test_*) ==="
BENCH=$(find "$RESULTS" -maxdepth 1 -type d \( -name "_bench*" -o -name "_ageprobe" -o -name "_test*" \) 2>/dev/null || true)
[[ -n "$BENCH" ]] && du -sh $BENCH || echo "(нет)"
if [[ $DELETE_BENCH -eq 1 ]]; then
    find "$RESULTS" -maxdepth 1 -type d \( -name "_bench*" -o -name "_ageprobe" -o -name "_test*" \) -exec rm -rf {} +
    echo "удалено"
else
    echo "(удаление: --delete-bench)"
fi

if [[ $DELETE_OLD -ge 0 ]]; then
    echo
    echo "=== прогоны старше $DELETE_OLD дн. ==="
    OLDER=$(find "$RESULTS" -maxdepth 1 -mindepth 1 -type d -mtime +"$DELETE_OLD" ! -name "_*" 2>/dev/null || true)
    if [[ -n "$OLDER" ]]; then
        du -sh $OLDER
        echo "УДАЛЯЕМ (это финальное подтверждение --delete-old):"
        echo "$OLDER" | while read -r d; do rm -rf "$d"; done
    else
        echo "(нет)"
    fi
fi

echo
echo "готово (без --delete-* ничего не удалено)"
