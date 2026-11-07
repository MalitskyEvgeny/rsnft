# Baseline и результаты ускорения forward-решателя

## Baseline (до оптимизаций, branch feat/rsnft3-uplift)

- Данные: Данные_для_прямой_задачи/data1, 60×120, max_dt 2.5 Ma, picard 3
- Эталонный прогон: results/d3fc48b257412100 (result.npz, метаданные с диагностикой)
- Замер (тёплый, -t 1): classic run_forward = 0.853 s (медиана 5 прогонов),
  81 шаг, avg picard 3.0, sum_linear_iters 11560
- Профиль: pcg_solve ≈ 60%+, сборка ≈ 25-30%, remap/прочее ≈ 5-10%
- Сравнение: julia --project=. benchmark_forward.jl --vs d3fc48b257412100

## Реализованные изменения (basin_fvm.jl)

1. Этап 1: Stencil (стенсил + CSC-паттерн строится один раз на шаг),
   FvmWorkspace (переиспользуемые буферы, double-buffer T), LithoPropLookup
   (массивы вместо Dict), picard_cycle на буферах, scatter без аллокаций.
   Classic-путь бит-в-бит равен baseline (max |ΔT| = 0.0).
2. Этап 2: linear_solver=:direct — CSC-матрица с фиксированным паттерном,
   символическая факторизация CHOLMOD кешируется по стенсилу
   (CHOLMOD_FACTOR_CACHE), численная — cholesky!, решение — ldiv!.
   Прямое решение точнее допуска (невязка ~5e-15), Пикар сходится за 2 итерации.
3. fallback :pcg сохранён (по умолчанию); ICC(0) не понадобился — вариант A выиграл.
4. П.15 (экстраполяция нач. приближения) и п.16 (инкрементальные свойства):
   не реализованы — при :direct Пикар уже на теоретическом минимуме (2 итерации),
   thermal_properties ≈ 5% времени профиля; выигрыш < 5% не окупает потерю
   бит-сходимости. П.17 (max_change по активным ячейкам) сделан в этапе 1.
5. П.18 (ослабление линейной точности) не применим к :direct (решение точное);
   для :pcg оставлена строгая схема как fallback.
6. run_inverse.jl/invert.jl: флаг --linear-solver (:pcg|:direct) прокидывается
   во все worker_init!*.
7. scripts/run_forward.sh: -O3 --check-bounds=no (отключается RSNFT_SAFE=1).

## Итог (60×120, data1, -t 1)

| Путь | Время | Ускорение |
|---|---|---|
| baseline classic | 0.853 s | 1× |
| classic + этап 1 | 0.85 s | ~1× (сборка < 25%, стенсил всё равно пересобирается на шаг) |
| param (геометрия) + этап 1 | 0.708 s | 1.2× |
| param + :direct | 0.563 s | 1.5× |
| param + :direct + кеш симв. факторизации | 0.379 s | **2.25×** |
| objective-оценка θ (30×60, обратная задача) | 166.7 → 69.8 ms | **2.39×** |

## Согласованность

- classic vs baseline: max |ΔT| = 0.0 (бит-в-бит), sum_linear_iters 11560 = baseline
- :direct vs :pcg: max |ΔT| = 2.2e-4 C, max |ΔRo| = 1.4e-7, |ΔJ|/J = 1.2e-5
- Тесты: test_core, test_fastpath (J бит-в-бит), test_adaptive, 
  test_adaptive_integration, test_objective, test_observations — PASS;
  новый test_direct.jl — PASS
