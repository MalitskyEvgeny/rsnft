# F8/F11: проверка входных папок/файлов до запуска тяжёлого пайплайна.
# Фатальные проблемы -> validate_inputs бросает ErrorException (caller делает exit(1));
# подозрительные, но допустимые конфигурации -> @warn.

isdefined(@__MODULE__, :XLSX) || import XLSX

const REQUIRED_FILES = isdefined(@__MODULE__, :REQUIRED) ? REQUIRED :
                       ("Depth.xlsx", "Lithology.xlsx", "Porosity.xlsx", "Tsurf.xlsx",
                        "lithotype_properties.xlsx")

"Фатальная проверка: data-dir существует и содержит обязательные входные файлы."
function validate_data_dir(data_dir::AbstractString)
    isempty(data_dir) && throw(ErrorException("не задан --data-dir"))
    isdir(data_dir) || throw(ErrorException("каталог данных не найден: $data_dir"))
    missing_files = [n for n in REQUIRED_FILES if !isfile(joinpath(data_dir, n))]
    isempty(missing_files) ||
        throw(ErrorException("в $data_dir отсутствуют обязательные файлы: " *
                             join(missing_files, ", ")))
    known = Set{String}(REQUIRED_FILES)
    for f in readdir(data_dir)
        endswith(f, ".xlsx") || continue
        f in known && continue
        # wells-файлы живут рядом с данными и используются по спецификации --wells
        occursin("well", lowercase(f)) && continue
        @warn "лишний xlsx в каталоге данных (будет проигнорирован)" file = f
    end
    return nothing
end

"Фатальная проверка wells-спецификации: файлы существуют и читаемы."
function validate_wells_spec(spec::AbstractString)
    isempty(spec) && throw(ErrorException("не задан --wells"))
    for path in split(spec, ";")
        path = strip(path)
        isempty(path) && throw(ErrorException("пустой элемент в --wells: \"$spec\""))
        isfile(path) || throw(ErrorException("wells-файл не найден: $path"))
        known_kinds = (".xlsx", ".json")
        any(endswith(lowercase(path), k) for k in known_kinds) ||
            throw(ErrorException("wells-файл должен быть .xlsx или .json: $path"))
        if endswith(lowercase(path), ".xlsx")
            local sheets
            try
                xf = XLSX.readxlsx(path)
                sheets = XLSX.sheetnames(xf)
            catch err
                throw(ErrorException("wells-файл не читается: $path ($(sprint(showerror, err)))"))
            end
            "Wells_Interp" in sheets ||
                @warn "в wells-файле нет листа Wells_Interp, будет использован первый лист" file = path
        end
    end
    return nothing
end

"Фатальная проверка каталога вывода: создаётся, права на запись есть."
function validate_out_dir(out_dir::AbstractString)
    isempty(out_dir) && throw(ErrorException("не задан --out"))
    try
        mkpath(out_dir)
        probe = joinpath(out_dir, ".write_probe")
        open(probe, "w") do io; print(io, "ok"); end
        rm(probe; force = true)
    catch err
        throw(ErrorException("каталог вывода не создаётся/нет прав записи: $out_dir " *
                             "($(sprint(showerror, err)))"))
    end
    old_cache = joinpath(out_dir, "j_cache.json")
    isfile(old_cache) && @warn "в каталоге вывода найден кэш старого формата j_cache.json (используется j_cache_v2.json)" file = old_cache
    return nothing
end

"Сводная валидация входов; при ошибке печатает понятное сообщение и exit(1)."
function validate_inputs(data_dir::AbstractString, wells::AbstractString, out_dir::AbstractString)
    try
        validate_data_dir(data_dir)
        isempty(wells) || validate_wells_spec(wells)
        validate_out_dir(out_dir)
    catch err
        println(stderr, "ОШИБКА КОНФИГУРАЦИИ: ", sprint(showerror, err))
        exit(1)
    end
    return nothing
end
