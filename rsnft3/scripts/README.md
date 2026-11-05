# run_adaptive — запуск адаптивного расчёта одной командой

Два эквивалентных скрипта-обёртки над `run_inverse.jl` (адаптивный бэкенд) +
`report_inverse.jl`:

| Скрипт | Платформа |
|---|---|
| `run_adaptive.ps1` | Windows PowerShell (5.1+) |
| `run_adaptive.sh` | Linux / macOS / любой POSIX shell (bash) |

После прогона печатается сводка приёмочных метрик из `solution.json` и пути артефактов.
Параметры у обоих совпадают (см. таблицу ниже); отличаются только имена флагов:
PowerShell `-Grid 30,60` ↔ bash `--grid 30,60`.

## Быстрый старт

Из корня репозитория (PowerShell):

```powershell
# контрольный прогон по умолчанию: поиск 20x40 + адаптивный контроль b16x32 L2
.\rsnft3\scripts\run_adaptive.ps1

# полностью адаптивный конвейер (поиск тоже на адаптивной сетке)
.\rsnft3\scripts\run_adaptive.ps1 -Mode search -AdaptiveBase 16,32 -AdaptiveLevel 1

# рабочий прогон 30x60 + адаптив 30x60 L1, отчёт открыть в браузере
.\rsnft3\scripts\run_adaptive.ps1 -Grid 30,60 -AdaptiveBase 30,60 -AdaptiveLevel 1 -OpenReport
```

При первом запуске, если политика выполнения запрещает скрипты:

```powershell
powershell -ExecutionPolicy Bypass -File .\rsnft3\scripts\run_adaptive.ps1
```

## Режимы

| Режим | Что делает |
|---|---|
| `-Mode final` (по умолч.) | поиск на равномерной сетке `-Grid`, затем адаптивный контроль при лучшем θ̂ (`--adaptive-final`) |
| `-Mode search` | весь конвейер на адаптивной сетке (`--adaptive-search`), равномерная сетка не используется |

## Основные параметры

Имена — в формате PowerShell; в bash те же имена с `--` и дефисами:
`-Grid` ↔ `--grid`, `-AdaptiveBase` ↔ `--adaptive-base`, `-SigmaT` ↔ `--sigma-T`,
`-MaxDtMa` ↔ `--max-dt-ma`, `-PcgTol` ↔ `--pcg-tol`, `-NoReport` ↔ `--no-report`,
`-FrozenMesh` ↔ `--frozen-mesh` и т.д.

| Параметр | По умолчанию | Смысл |
|---|---|---|
| `-Grid` | `20,40` | равномерная сетка поиска nx,nz |
| `-AdaptiveBase` | `16,32` | базовый грид quadtree |
| `-AdaptiveLevel` | `2` | макс. уровень измельчения (0..5) |
| `-AdaptivePreview` | `120,160` | растеризация для выборки наблюдений |
| `-Theta0File` | — | `solution.json` для полировки с готового θ̂ (DE пропускается) |
| `-Ensemble` / `-Workers` | 1 / 1 | число стартов DE / воркеров Julia |
| `-DeBudget` / `-NmIterations` | 2000 / 30 | бюджет DE / итерации NM |
| `-Reg` | 1.0 | вес регуляризации (для twin ставьте 0) |
| `-Seed` | 1 | seed |
| `-SigmaT` / `-SigmaRo` | 2.0 / 0.05 | СКВО шума наблюдений |
| `-MaxDtMa` | 2.5 | шаг решателя, Ma |
| `-PcgTol` | `1e-4` | точность PCG (главный рычаг скорости) |
| `-Threads` | 8 | потоки Julia (`-t`) |
| `-FrozenMesh` | выкл | эксперимент.: меш строится один раз |
| `-OutDir` | `results/adaptive_<mode>_<время>` | каталог результата |
| `-DataDir` / `-Wells` | автоподстановка | каталог данных / список XLSX через `;` |
| `-NoReport` | выкл | не генерировать report.html |
| `-OpenReport` | выкл | открыть отчёт в браузере после прогона |

## Что получается на выходе

В `-OutDir`:

- `solution.json` — лучшее θ̂, breakdown J, критерии, блок `adaptive`;
- `result.npz` (и `result_adaptive.npz` в режиме `-Mode search`) — поля прогона;
- `mesh_adaptive.png` — картинка адаптивного меша с температурой;
- `report.html` — HTML-отчёт (если не указан `-NoReport`).

В консоли в конце печатается сводка:

```
=== summary ===
status   : ok
rmse_T   : 3.44   (accept: <= 4 = 2*sigma_T)
rmse_Ro  : 0.031  (accept: <= 0.10)
T in 2s  : 96.3%  (accept: >= 95%)
adaptive : base=16,32 level=2  rmse_T=3.40  ...
theta    : 105.3, 118.9, ... , 59.8
mesh     : ...\results\adaptive_final_20260101_120000\mesh_adaptive.png
total    : 00:03:12
```

## Типовые сценарии

```powershell
# 1. Быстрая разведка (~2-3 мин)
.\rsnft3\scripts\run_adaptive.ps1

# 2. Полировка с готового θ̂ + адаптивный контроль
.\rsnft3\scripts\run_adaptive.ps1 -Theta0File results\inverse_final\solution.json -NmIterations 10

# 3. Twin-проверка на адаптиве (важно: -Reg 0)
.\rsnft3\scripts\run_adaptive.ps1 -Mode search -Reg 0 -Wells results\twin\twin.json

# 4. Параллельный поиск (воркеры окупаются при >= 0.15 c/оценку, т.е. на 30x60 и крупнее)
.\rsnft3\scripts\run_adaptive.ps1 -Grid 30,60 -Ensemble 4 -Workers 4 -DeBudget 6000
```

## Замечания

- Каталог данных по умолчанию: `<repo>/Данные для обратной задачи`; колодцы —
  `TemperatureWellData.xlsx;VitriniteWellData.xlsx`. Переопределяется `-DataDir` / `-Wells`.
- Скрипт проверяет наличие `rsnft3/Project.toml`, `julia` на PATH, каталога данных и
  `-Theta0File` до запуска; ненулевой код возврата Julia прерывает выполнение.
- Дробные параметры передаются в Julia через InvariantCulture (ps1) / как есть (sh) —
  скрипты безопасны при любой региональной настройке.
- `results/` в `.gitignore`: выводы фиксируйте текстом коммитов, а не файлами.
- Скрипты не пушат и не коммитят ничего сами.
- В Windows Git-Bash запускать не рекомендуется: Julia-бинарник Windows не понимает
  `/c/...`-пути; на Windows используйте `run_adaptive.ps1`, на Linux — `run_adaptive.sh`.
