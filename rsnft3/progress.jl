# Однострочный прогресс-бар в консоли: точная полоса (известен итог), подпись
# этапа/шага на английском, fallback на вехи при не-TTY. Ноль зависимостей.
#
# Гарантии «не мешать и не замедлять»:
#   * рисует только главный процесс (myid() == 1);
#   * троттлинг: перерисовка не чаще PROGRESS_MIN_INTERVAL (0.5 с), проверка —
#     дешёвое сравнение time(), мгновенно только на смене этапа/завершении;
#   * рендер — одна строка в stdout (stderr-вывод ломает PowerShell 5 при 2>&1), весь рендер в
#     try/catch: ошибка вывода никогда не роняет расчёт;
#   * полностью отключаемо: env RSNFT_PROGRESS=off (--no-progress в run_inverse)
#     или не-TTY stderr — обновления становятся чистым no-op.

module Progress

using Printf
using Distributed

export stage_set, bar_start, bar_add, bar_finish, step_note, plog, init,
       fmt_time_public,
       progress_enable!, progress_disable!, progress_enabled

const MIN_INTERVAL = 0.5          # сек между перерисовками TTY-полосы
const MILESTONE_MIN_INTERVAL = 10.0  # сек между вехами в не-TTY
const MILESTONE_MIN_STEP = 0.05   # мин доля прогресса между вехами
const WIDTH = 20                  # ширина полосы в символах

mutable struct Bar
    total::Float64
    value::Float64
    caption::String
    t0::Float64
end

mutable struct BarState
    enabled::Bool
    tty::Bool
    stage_k::Int
    stage_n::Int
    stage_name::String
    active::Bool
    total::Float64
    value::Float64
    caption::String
    note::String
    t0::Float64
    last_render::Float64
    last_milestone::Float64
    last_milestone_value::Float64
    workers::Dict{Int,Bar}          # per-worker bars (pid -> Bar)
    worker_order::Vector{Int}       # stable row order
end

const STATE = BarState(false, false, 0, 0, "", false, 0.0, 0.0, "", "", 0.0, 0.0, 0.0, -1.0,
                      Dict{Int,Bar}(), Int[])

"Инициализация по окружению: TTY stderr, env RSNFT_PROGRESS. Вызывать на старте main."
function init()
    tty = try
        stdout isa Base.TTY
    catch
        false
    end
    STATE.tty = tty
    off_env = lowercase(get(ENV, "RSNFT_PROGRESS", "")) in ("off", "0", "no", "none")
    STATE.enabled = !off_env          # off=полное отключение; не-TTY -> вехи, TTY -> полоса
    return STATE.enabled
end

progress_enabled() = STATE.enabled
progress_enable!() = (STATE.enabled = true; nothing)
progress_disable!() = (STATE.enabled = false; nothing)

"Локальный воркер/не-главный процесс — обновления всегда no-op."
@inline active() = STATE.enabled && myid() == 1

fmt_time(s) = @sprintf("%02d:%02d", floor(Int, s / 60), floor(Int, s) % 60)

fmt_time_public(s) = fmt_time(s)

"Смена этапа всего прогона: мгновенная перерисовка."
function stage_set(k::Int, N::Int, name::AbstractString)
    STATE.stage_k, STATE.stage_n, STATE.stage_name = k, N, String(name)
    active() || return nothing
    if STATE.tty
        print(stdout, "\r\033[K=== Stage $k/$N: $name ===\n")
    else
        println(stdout, "=== Stage $k/$N: $name ===")
    end
    STATE.last_render = time()
    return nothing
end

"Открыть точную полосу: всего шагов total, подпись caption."
function bar_start(total::Real, caption::AbstractString)
    STATE.active = true
    STATE.total = Float64(total)
    STATE.value = 0.0
    STATE.caption = String(caption)
    STATE.note = ""
    STATE.t0 = time()
    STATE.last_milestone_value = 0.0
    active() || return nothing
    STATE.last_render = time()
    STATE.last_milestone = STATE.last_render
    STATE.tty && render()
    return nothing
end

"Добавить n шагов; обновляет полосу с троттлингом. Дешёвый вызов: счётчик + time()."
@inline function bar_add(n::Real = 1, caption::Union{Nothing,AbstractString} = nothing)
    (STATE.enabled && myid() == 1) || return nothing
    STATE.value += n
    caption === nothing || (STATE.caption = caption)
    active() || return nothing
    now = time()
    if STATE.tty
        now - STATE.last_render >= MIN_INTERVAL && render(now)
    else
        frac = STATE.total > 0 ? STATE.value / STATE.total : 0.0
        if frac - STATE.last_milestone_value >= MILESTONE_MIN_STEP &&
           now - STATE.last_milestone >= MILESTONE_MIN_INTERVAL
            STATE.last_milestone = now
            STATE.last_milestone_value = frac
            milestone()
        end
    end
    return nothing
end

"Закрыть полосу: мгновенно дорисовать 100% и перевести строку."
function bar_finish()
    if active()
        if STATE.tty
            STATE.value = STATE.total
            render(time())
            n = length(STATE.worker_order)
            n > 0 && print(stdout, repeat("[1B[K", n), repeat("[1A", n), "\r[K")
            empty!(STATE.workers)
            empty!(STATE.worker_order)
            print(stdout, "\n")
        else
            frac = STATE.total > 0 ? 1.0 : 0.0
            STATE.last_milestone_value = frac
            STATE.last_milestone = time()
            println(stdout, @sprintf("[100%%] %s done · %s elapsed", STATE.caption,
                                     fmt_time(time() - STATE.t0)))
        end
    end
    STATE.active = false
    return nothing
end

"Подпись без процентов (полировка, вехи ансамбля): мгновенная перерисовка строки."
function step_note(caption::AbstractString)
    STATE.note = String(caption)
    active() || return nothing
    if STATE.tty
        now = time()
        now - STATE.last_render >= MIN_INTERVAL || return nothing
        STATE.last_render = now
        render(now)
    else
        println(stdout, "  Step: $(STATE.note)")
    end
    return nothing
end

"Бар одного воркера: создаётся при первом рапорте."
function worker_bar_update(pid::Int, value::Real, total::Real, caption::AbstractString)
    (STATE.enabled && myid() == 1) || return nothing
    b = get!(STATE.workers, pid, Bar(Float64(total), Float64(value), String(caption), time()))
    if !(pid in STATE.worker_order)
        push!(STATE.worker_order, pid)
    end
    b.value = Float64(value)
    b.total = Float64(total)
    b.caption = String(caption)
    STATE.tty && render_all()
    return nothing
end

"Воркер завершил старт: бар заполняется и метится done."
function worker_bar_done(pid::Int)
    (STATE.enabled && myid() == 1) || return nothing
    haskey(STATE.workers, pid) || return nothing
    b = STATE.workers[pid]
    b.value = b.total
    b.caption = b.caption * " · done"
    STATE.tty && render_all()
    return nothing
end

"Сброс воркер-баров перед новой стадии ансамбля."
function worker_bars_clear()
    (STATE.enabled && myid() == 1) || return nothing
    if STATE.tty && !isempty(STATE.workers)
        n = length(STATE.worker_order)
        print(stdout, repeat("[1B[K", n), repeat("[1A", n))
    end
    empty!(STATE.workers)
    empty!(STATE.worker_order)
    return nothing
end

"Напечатать обычное сообщение, не ломая полосу: очистка строки, msg, перерисовка."
function plog(msg::AbstractString)
    if myid() != 1
        println(msg)
        return nothing
    end
    if STATE.tty && STATE.active
        print(stdout, "\r\033[K", msg, "\n")
        STATE.last_render = time()
        render(STATE.last_render)
    else
        println(msg)
    end
    return nothing
end

function bar_line(now::Float64)
    frac = STATE.total > 0 ? clamp(STATE.value / STATE.total, 0.0, 1.0) : 0.0
    filled = round(Int, frac * WIDTH)
    bar = repeat("█", filled) * repeat("░", WIDTH - filled)
    elapsed = now - STATE.t0
    eta = frac > 0.01 ? @sprintf("~%s left", fmt_time(elapsed / frac - elapsed)) : "estimating…"
    extra = isempty(STATE.note) ? "" : " · " * STATE.note
    return @sprintf("[%s] %3d%% · %s · %s elapsed · %s%s",
                    bar, round(Int, frac * 100), STATE.caption,
                    fmt_time(elapsed), eta, extra)
end

const CSI_CLEAR = "\r[K"

"Рисование всех воркер-баров (одна строка на воркер), без перемещения курсора."
function render_all()
    try
        print(stdout, CSI_CLEAR)
        for pid in STATE.worker_order
            b = STATE.workers[pid]
            frac = b.total > 0 ? clamp(b.value / b.total, 0.0, 1.0) : 0.0
            filled = round(Int, frac * WIDTH)
            bar = repeat("█", filled) * repeat("░", WIDTH - filled)
            el = b.t0 > 0 ? time() - b.t0 : 0.0
            print(stdout, @sprintf("  worker %-2d [%s] %3d%% · %s · %s", 
                                   pid, bar, round(Int, frac * 100), b.caption, fmt_time(el)))
        end
        flush(stdout)
    catch
        progress_disable!()
    end
    return nothing
end


"Строки воркеров под главной: cursor-down + clear, без 
 (один блок с главной строкой)."
function render_worker_rows()
    n = length(STATE.worker_order)
    n == 0 && return nothing
    for pid in STATE.worker_order
        b = STATE.workers[pid]
        frac = b.total > 0 ? clamp(b.value / b.total, 0.0, 1.0) : 0.0
        filled = round(Int, frac * WIDTH)
        bar = repeat("█", filled) * repeat("░", WIDTH - filled)
        el = b.t0 > 0 ? time() - b.t0 : 0.0
        print(stdout, "[1B[K", @sprintf("  worker %-2d [%s] %3d%% · %s · %s", 
                               pid, bar, round(Int, frac * 100), b.caption, fmt_time(el)))
    end
    print(stdout, repeat("[1A", n))
    return nothing
end
function render(now::Float64 = time())
    try
        print(stdout, "\r\033[K", bar_line(now))
        !isempty(STATE.worker_order) && render_worker_rows()
        flush(stdout)
    catch
        progress_disable!()
    end
    STATE.last_render = now
    return nothing
end

function milestone()
    frac = STATE.total > 0 ? clamp(STATE.value / STATE.total, 0.0, 1.0) : 0.0
    println(stdout, @sprintf("[%3d%%] %s · %s elapsed",
                             round(Int, frac * 100), STATE.caption,
                             fmt_time(time() - STATE.t0)))
    return nothing
end

end # module
