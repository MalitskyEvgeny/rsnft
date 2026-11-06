# Скрипты запуска и HTML-отчёт

## Что здесь

| Скрипт | Windows | Linux / macOS | Что делает |
|---|---|---|---|
| Прямая задача + отчёт | `run_forward.ps1` | `run_forward.sh` | прогон солвера → отчёт секции «Прямая задача» |
| Прямая + обратная + отчёт | `run_full.ps1` | `run_full.sh` | прямой прогон + инверсия `run_inverse.jl` → единый отчёт |
| Адаптивная инверсия | `run_adaptive.ps1` | `run_adaptive.sh` | прежний сценарий (адаптивный бэкенд инверсии), отчёт — секция «Обратная задача» |

**Основной режим — адаптивная сетка.** Прямая задача считается адаптивным
солвером (`adaptive_basin_fvm.jl`, quadtree-измельчение у границ слоёв),
инверсия в `run_full` включает контроль на адаптивной сетке
(`run_inverse.jl --adaptive-final`). Флаг `--uniform` / `-Uniform`
переключает на классические равномерные сетки (`basin_fvm.jl`).

Отчёт собирает **`rsnft3/report_full.jl`** — один самодостаточный HTML-файл
на русском языке с интерактивными графиками (Plotly встроен в файл, отчёт
открывается офлайн, без интернета). Состав секций зависит от флагов:
`--direct` → секция «Прямая задача», `--inverse` → секция «Обратная задача»,
оба флага → один общий отчёт с оглавлением.

## Требования

- Julia 1.9+ (проверено на 1.13), окружение `rsnft3/Project.toml`
  (`julia --project=rsnft3`).
- Дополнительных пакетов не нужно: plotly.min.js лежит в `rsnft3/report/`
  и встраивается в отчёт при генерации.
- Папки с данными в корне репозитория (как есть сейчас):
  - `Данные_для_прямой_задачи/` — варианты `data1`, `data2` (XLSX прямой задачи);
  - `Данные для обратной задачи/` — скважинные данные
    (`TemperatureWellData.xlsx`, `VitriniteWellData.xlsx`, …).

Если папки лежат в другом месте — передайте их флагами (`--data-dir`,
`--inv-data-dir`, `--wells`), ничего перемещать не нужно.

## run_forward — прямая задача с отчётом

PowerShell (из корня репозитория):

```powershell
.\rsnft3\scripts\run_forward.ps1                        # адаптивная сетка (base 16x32, level 2)
.\rsnft3\scripts\run_forward.ps1 -AdaptiveBase 30,60 -AdaptiveLevel 1
.\rsnft3\scripts\run_forward.ps1 -Uniform -Nx 30 -Nz 60 -Force  # равномерная 30x60
.\rsnft3\scripts\run_forward.ps1 -DataDir "C:\путь\к\данным" -OpenReport
```

Bash:

```bash
./rsnft3/scripts/run_forward.sh                        # адаптивная сетка (base 16x32, level 2)
./rsnft3/scripts/run_forward.sh --adaptive-base 30,60 --adaptive-level 1
./rsnft3/scripts/run_forward.sh --uniform --nx 30 --nz 60 --force
./rsnft3/scripts/run_forward.sh --data-dir ../Данные_для_прямой_задачи/data2 --open
```

Флаги (PowerShell ↔ bash): `-DataDir/--data-dir`,
`-AdaptiveBase/--adaptive-base` (базовая сетка адаптивного солвера, `16,32`),
`-AdaptiveLevel/--adaptive-level` (уровень измельчения, `2`),
`-Uniform/--uniform`, `-Nx/--nx`, `-Nz/--nz` (только с `-Uniform/--uniform`),
`-MaxDtMa/--max-dt-ma`, `-PicardIters/--picard-iters`, `-Out/--out`,
`-Force/--force`, `-OpenReport/--open`.

Что получается:
- `results/<run_id>/result.npz`, `metadata.json` — кэш прогона солвера;
- `results/report_forward_<время>/report.html` — отчёт.

## run_full — прямая + обратная задачи с общим отчётом

PowerShell:

```powershell
.\rsnft3\scripts\run_full.ps1 -Grid 30,60 -DeBudget 200    # быстрый контрольный прогон
.\rsnft3\scripts\run_full.ps1 -Twin -Q 75,80,85,90         # twin-режим
.\rsnft3\scripts\run_full.ps1 -Ensemble 4 -Workers 4 -DeBudget 8000  # рабочий прогон
```

Bash:

```bash
./rsnft3/scripts/run_full.sh --grid 30,60 --de-budget 200
./rsnft3/scripts/run_full.sh --twin --q 75,80,85,90
./rsnft3/scripts/run_full.sh --ensemble 4 --workers 4 --de-budget 8000
```

Флаги (PowerShell ↔ bash):

| PowerShell | Bash | По умолчанию | Смысл |
|---|---|---|---|
| `-DataDir` | `--data-dir` | `<repo>/Данные_для_прямой_задачи/data1` | данные прямой задачи |
| `-InvDataDir` | `--inv-data-dir` | `<repo>/Данные для обратной задачи` | данные для инверсии |
| `-Wells` | `--wells` | Temperature+Vitrinite XLSX из `--inv-data-dir` | скважины, через `;` |
| `-Twin` | `--twin` | выкл | сгенерировать twin-наблюдения перед инверсией |
| `-Q` | `--q` | `75,80,85,90` | истинный q для twin (значение на эпоху) |
| `-Grid` | `--grid` | `20,40` | рабочая сетка инверсии `nx,nz` |
| `-Base` | `--base` | `16,32` | адаптивная база прямой задачи |
| `-Level` | `--level` | `2` | адаптивный уровень прямой задачи |
| `-AdaptiveBase` | `--adaptive-base` | `16,32` | адаптивная база инверсии |
| `-AdaptiveLevel` | `--adaptive-level` | `2` | адаптивный уровень инверсии |
| `-Uniform` | `--uniform` | выкл | равномерные сетки вместо адаптивных |
| `-Nx` / `-Nz` | `--nx` / `--nz` | `60` / `120` | сетка равномерной прямой задачи (только с `-Uniform`) |
| `-Ensemble` | `--ensemble` | `1` | число стартов ансамбля |
| `-Workers` | `--workers` | `1` | параллельные воркеры |
| `-DeBudget` | `--de-budget` | `2000` | бюджет оценок DE |
| `-NmIterations` | `--nm-iterations` | `30` | итерации NM-полировки |
| `-SigmaT` / `-SigmaRo` | `--sigma-T` / `--sigma-Ro` | `2.0` / `0.05` | шумы наблюдений |
| `-Reg` | `--reg` | `1.0` | вес регуляризации |
| `-Seed` | `--seed` | `1` | зерно генератора |
| `-Theta0File` | `--theta0-file` | нет | solution.json для полировки без DE |
| `-Out` | `--out` | `results/report_full_<время>` | каталог результатов |
| `-Force` | `--force` | выкл | пересчитать прямой прогон из кэша |
| `-OpenReport` | `--open` | выкл | открыть отчёт в браузере |

Что получается в `results/report_full_<время>/`:
- `report.html` — единый отчёт (обе секции + оглавление);
- `inverse/solution.json`, `inverse/result.npz` — результат инверсии;
- `twin_grid.json` (в twin-режиме) — сгенерированные наблюдения.

## Отчёт напрямую (без скриптов)

```bash
cd rsnft3
julia --project=. report_full.jl --direct ../results/<run_id>            # только прямая
julia --project=. report_full.jl --inverse ../results/<inverse_run>      # только обратная
julia --project=. report_full.jl --direct ... --inverse ... --out my.html  # общий
```

Каталог `--direct` должен содержать `result.npz` (и `metadata.json`);
каталог `--inverse` — `solution.json` (и `result.npz`).

## Содержимое отчёта

- **Прямая задача**: сводка прогона (сетка/адаптивная база, шаги, время, min/max T);
  температурное поле T(x,z) со слайдером по эпохам; термические профили
  в колонках; геометрия бассейна (поверхность/кровля фундамента);
  литология; пористость.
- **Обратная задача**: вердикт по условию 18 (RMSE ≤ 2σ, ≥95% точек в 2σ);
  восстановленный поток q(x,t) heatmap + таблица; профили T и Ro по скважинам
  (модель против наблюдений с ±σ); ансамбль решений (разброс θ, вклады
  в функционал); настройки прогона.

## Ошибки запуска PowerShell

При первом запуске, если политика выполнения запрещает скрипты:

```powershell
powershell -ExecutionPolicy Bypass -File .\rsnft3\scripts\run_full.ps1
```
