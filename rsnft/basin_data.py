"""
Читалка данных для задачи Академического турнира 2026:
двумерная модель теплопереноса в осадочном чехле.

Загружает 6 xlsx-файлов и предоставляет удобный API для доступа
к геометрии, литологии, пористости, граничным условиям и свойствам литотипов.

Использование:
    from basin_data import BasinData

    data = BasinData("C:/path/to/xlsx_dir")

    # Временные шаги и слои
    data.times           # [0.0, 5.0, 10.0, ..., 200.0]
    data.layers          # ['Layer_1', 'Layer_2', ..., 'Layer_7']

    # Геометрия слоя в момент времени
    x, z_top, z_bot = data.geometry('Layer_5', time=75.0)

    # Литотип, пористость
    lith_codes = data.lithology('Layer_5', time=75.0)   # массив кодов
    porosity   = data.porosity('Layer_5', time=75.0)     # массив φ (в %)

    # Граничные условия
    x, tsurf   = data.tsurf(time=0.0)         # Tsurf(x) [°C]
    x, qbasal  = data.heat_flow(time=0.0)     # qbasal(x) [mW/m²]

    # Свойства литотипов
    data.lithotypes
    # {1001: {'name': 'Halite_new', 'rho_s': 2200.0, 'lambda_20': 6.5,
    #         'c20_J': 862.5, 'As': 0.0124, ...}, ...}

    # Какие слои существуют в данный момент
    data.layers_at(time=75.0)  # ['Layer_5', 'Layer_5_1', ..., 'Layer_7']

    # Кинетика EASY%Ro (Sweeney & Burnham, 1990)
    data.easy_ro  # {'A': [...], 'E': [...], 'f': [...]}
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


# -----------------------------------------------------------------------
# Кинетические константы EASY%Ro (Sweeney & Burnham, 1990)
# -----------------------------------------------------------------------

_EASY_RO = {
    # Энергии активации Ei [ккал/моль]
    'E_kcal': np.array([
        34, 36, 38, 40, 42, 44, 46, 48, 50, 52,
        54, 56, 58, 60, 62, 64, 66, 68, 70, 72,
    ], dtype=np.float64),

    # Весовые доли fi
    'f': np.array([
        0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.04, 0.04, 0.07,
        0.06, 0.06, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01,
    ], dtype=np.float64),

    # Предэкспоненциальный множитель A [1/с] — единый для всех реакций
    'A': 1.0e13,

    # Универсальная газовая постоянная Rg [ккал/(моль·К)]
    'Rg_kcal': 1.987e-3,
}

# Перевод в СИ
_EASY_RO['E_J'] = _EASY_RO['E_kcal'] * 4186.8     # Дж/моль
_EASY_RO['Rg_J'] = 8.314                            # Дж/(моль·К)


# -----------------------------------------------------------------------
# Вспомогательные функции
# -----------------------------------------------------------------------

def _layer_sort_key(name):
    """Ключ сортировки: 'Layer_4_1' -> (4, 1)."""
    parts = name.replace('Layer_', '').split('_')
    return tuple(int(p) for p in parts)


def _load_property_xlsx(filepath):
    """
    Загружает xlsx со свойством.
    Возвращает dict: layer_name -> DataFrame.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Файл не найден: {filepath}\n"
            f"В Linux критически важен регистр букв (например, 'Depth.xlsx', а не 'depth.xlsx')."
        )
    print(f"Читаю {filepath}")
    xls = pd.ExcelFile(filepath)
    result = {}
    for sheet in xls.sheet_names:
        result[sheet] = pd.read_excel(xls, sheet)
    return result


# -----------------------------------------------------------------------
# Модели Секигучи и Ваплеса
# -----------------------------------------------------------------------

def sekiguchi(T_K, lambda_20):
    """
    Теплопроводность скелета λs(T) по модели Sekiguchi (1984).

    Параметры:
        T_K       : температура [К]
        lambda_20 : теплопроводность при 20°C [Вт/(м·К)]

    Возвращает:
        λs [Вт/(м·К)]
    """
    return 1.84 + 358.0 * (1.0227 * lambda_20 - 1.882) * (1.0 / T_K - 0.00068)


def waples(T_K, c20_J):
    """
    Удельная теплоёмкость скелета cs(T) по модели Waples (2004).

    Параметры:
        T_K   : температура [К]
        c20_J : теплоёмкость при 20°C [Дж/(кг·К)]

    Возвращает:
        cs [Дж/(кг·К)]
    """
    Tc = T_K - 273.15
    return c20_J * (0.953 + 2.29e-3 * Tc - 2.835e-6 * Tc**2 + 1.191e-9 * Tc**3)


# -----------------------------------------------------------------------
# Эффективные свойства среды
# -----------------------------------------------------------------------

def effective_lambda(T_K, lambda_20, phi, lambda_f=0.7):
    """
    Эффективная теплопроводность: λ = λs^(1-φ) · λf^φ  (геом. среднее).

    Параметры:
        T_K       : температура [К]
        lambda_20 : теплопроводность скелета при 20°C [Вт/(м·К)]
        phi       : пористость (доля, 0..1)
        lambda_f  : теплопроводность флюида [Вт/(м·К)], по умолч. 0.7
    """
    ls = sekiguchi(T_K, lambda_20)
    return ls**(1.0 - phi) * lambda_f**phi


def effective_rho_c(T_K, c20_J, rho_s, phi,
                    rho_f=1040.0, c_f=4184.0):
    """
    Эффективная объёмная теплоёмкость: (ρc) = (1-φ)ρs·cs + φ·ρf·cf.

    Параметры:
        T_K   : температура [К]
        c20_J : теплоёмкость скелета при 20°C [Дж/(кг·К)]
        rho_s : плотность скелета [кг/м³]
        phi   : пористость (доля, 0..1)
        rho_f : плотность флюида [кг/м³]
        c_f   : теплоёмкость флюида [Дж/(кг·К)]
    """
    cs = waples(T_K, c20_J)
    return (1.0 - phi) * rho_s * cs + phi * rho_f * c_f


def effective_A(As, phi):
    """
    Эффективное радиогенное тепловыделение: A = (1-φ)·As.

    Параметры:
        As  : тепловыделение скелета [µW/m³]
        phi : пористость (доля, 0..1)
    """
    return (1.0 - phi) * As


# -----------------------------------------------------------------------
# Основной класс
# -----------------------------------------------------------------------

class BasinData:
    """Контейнер данных бассейна для задачи теплопереноса."""

    def __init__(self, xlsx_dir, lithotype_file=None):
        """
        Параметры:
            xlsx_dir       : папка с Depth.xlsx, Lithology.xlsx, Porosity.xlsx,
                             Heat_Flow_Basal.xlsx, Tsurf.xlsx
            lithotype_file : путь к lithotype_properties.xlsx
                             (по умолчанию ищет в xlsx_dir)
        """
        self._dir = Path(xlsx_dir).expanduser().resolve()

        if not self._dir.exists():
            raise FileNotFoundError(f"Директория с данными не найдена: {self._dir}")

        # Загрузка свойств-полей
        self._depth = _load_property_xlsx(self._dir / 'Depth.xlsx')
        self._lith = _load_property_xlsx(self._dir / 'Lithology.xlsx')
        self._poro = _load_property_xlsx(self._dir / 'Porosity.xlsx')
        self._hf = _load_property_xlsx(self._dir / 'Heat_Flow_Basal.xlsx')
        self._tsurf = _load_property_xlsx(self._dir / 'Tsurf.xlsx')

        # Слои (общий список из Depth)
        self.layers = sorted(self._depth.keys(), key=_layer_sort_key)

        # Временные шаги (объединение всех)
        all_times = set()
        self._layer_times = {}
        for layer, df in self._depth.items():
            ts = sorted(df.iloc[:, 1].unique())
            self._layer_times[layer] = ts
            all_times.update(ts)
        self.times = sorted(all_times)

        # Свойства литотипов
        if lithotype_file is None:
            lithotype_file = self._dir / 'lithotype_properties.xlsx'
        else:
            lithotype_file = Path(lithotype_file).expanduser().resolve()
        self.lithotypes = self._load_lithotypes(lithotype_file)

        # Кинетика EASY%Ro
        self.easy_ro = _EASY_RO.copy()

    # -------------------------------------------------------------------
    # Загрузка литотипов
    # -------------------------------------------------------------------

    @staticmethod
    def _load_lithotypes(filepath):
        """Загружает lithotype_properties.xlsx в словарь {ID: {...}}."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Файл литотипов не найден: {filepath}")
        df = pd.read_excel(filepath)
        result = {}
        for _, row in df.iterrows():
            lid = int(row.iloc[0])
            result[lid] = {
                'name':      row.iloc[1],
                'rho_s':     float(row.iloc[2]),     # кг/м³
                'lambda_20': float(row.iloc[3]),     # Вт/(м·К)
                'c20_kcal':  float(row.iloc[4]),     # ккал/(кг·К)
                'c20_J':     float(row.iloc[5]),     # Дж/(кг·К)
                'U_ppm':     float(row.iloc[6]),
                'Th_ppm':    float(row.iloc[7]),
                'K_pct':     float(row.iloc[8]),
                'As':        float(row.iloc[9]),     # µW/m³
            }
        return result

    # -------------------------------------------------------------------
    # Доступ к данным
    # -------------------------------------------------------------------

    def layers_at(self, time):
        """Список слоёв, существующих в момент time (отсортировано сверху вниз)."""
        return [l for l in self.layers if time in self._layer_times.get(l, [])]

    def _get_slice(self, storage, layer, time):
        """Извлекает строки для (layer, time) из хранилища."""
        df = storage.get(layer)
        if df is None:
            raise KeyError(f'Слой {layer} не найден')
        sub = df[df.iloc[:, 1] == time]
        if len(sub) == 0:
            raise KeyError(
                f'Слой {layer} не существует в момент {time} Ma'
            )
        return sub

    def geometry(self, layer, time):
        """
        Геометрия слоя.

        Возвращает:
            x     : np.array — координата вдоль разреза [м]
            z_top : np.array — глубина кровли [м]
            z_bot : np.array — глубина подошвы [м]
        """
        sub = self._get_slice(self._depth, layer, time)
        return (sub.iloc[:, 0].values.astype(np.float64),
                sub.iloc[:, 2].values.astype(np.float64),
                sub.iloc[:, 3].values.astype(np.float64))

    def lithology(self, layer, time):
        """Коды литотипов (Aver). Возвращает np.array of int."""
        sub = self._get_slice(self._lith, layer, time)
        return sub.iloc[:, 7].values.astype(int)

    def porosity(self, layer, time):
        """
        Пористость (Aver).

        Возвращает:
            φ : np.array — пористость в **процентах**
                Для формул (2)–(4) делите на 100.
        """
        sub = self._get_slice(self._poro, layer, time)
        return sub.iloc[:, 7].values.astype(np.float64)

    def tsurf(self, time):
        """
        Температура поверхности Tsurf(x, t) — граничное условие (8).

        Возвращает:
            x     : np.array [м]
            tsurf : np.array [°C]

        Берётся из самого верхнего слоя в момент time.
        """
        top_layer = self.layers_at(time)[0]
        sub = self._get_slice(self._tsurf, top_layer, time)
        return (sub.iloc[:, 0].values.astype(np.float64),
                sub.iloc[:, 7].values.astype(np.float64))

    def heat_flow(self, time):
        """
        Базальный тепловой поток qbasal(x, t) — граничное условие (9).

        Возвращает:
            x      : np.array [м]
            qbasal : np.array [мВт/м²]

        Берётся из самого нижнего слоя в момент time.
        """
        bot_layer = self.layers_at(time)[-1]
        sub = self._get_slice(self._hf, bot_layer, time)
        return (sub.iloc[:, 0].values.astype(np.float64),
                sub.iloc[:, 7].values.astype(np.float64))

    def get_lithotype(self, code):
        """Свойства литотипа по коду. Возвращает dict."""
        if code not in self.lithotypes:
            raise KeyError(f'Литотип {code} не найден в библиотеке')
        return self.lithotypes[code]

    # -------------------------------------------------------------------
    # Полная конфигурация на момент времени
    # -------------------------------------------------------------------

    def snapshot(self, time):
        """
        Полный снимок бассейна в момент time.

        Возвращает list of dict (сверху вниз), каждый элемент:
            {
                'layer':    str,
                'x':        np.array,
                'z_top':    np.array,
                'z_bot':    np.array,
                'lith':     np.array (int),
                'porosity': np.array (%, делить на 100 для формул),
            }
        """
        result = []
        for layer in self.layers_at(time):
            x, zt, zb = self.geometry(layer, time)
            result.append({
                'layer': layer,
                'x': x,
                'z_top': zt,
                'z_bot': zb,
                'lith': self.lithology(layer, time),
                'porosity': self.porosity(layer, time),
            })
        return result

    # -------------------------------------------------------------------
    # Информация
    # -------------------------------------------------------------------

    def __repr__(self):
        return (
            f'BasinData(\n'
            f'  layers:    {len(self.layers)} ({self.layers[0]} .. {self.layers[-1]})\n'
            f'  times:     {len(self.times)} шагов '
            f'({self.times[0]:g} .. {self.times[-1]:g} Ma)\n'
            f'  lithotypes: {list(self.lithotypes.keys())}\n'
            f')'
        )


# -----------------------------------------------------------------------
# Пример использования
# -----------------------------------------------------------------------

if __name__ == '__main__':
    # ========== НАСТРОЙКИ ==========
    # XLSX_DIR = r"C:\Users\DrWhite\Desktop\turnir_data_26\data1"
    # XLSX_DIR = r"C:\Users\DrWhite\Desktop\turnir_data_26\data2"
    XLSX_DIR = Path(__file__).parent / "data1"
    # ===============================

    data = BasinData(XLSX_DIR)
    print(data)

    # Пример: конфигурация на 0 Ma
    print('\n--- Снимок бассейна на 0 Ma ---')
    snap = data.snapshot(0.0)
    for s in snap:
        lith_codes = np.unique(s['lith'])
        names = [data.lithotypes[c]['name'] for c in lith_codes]
        print(f"  {s['layer']:15s}  x: {s['x'][0]:.0f}..{s['x'][-1]:.0f} m  "
              f"z: {s['z_top'].min():.0f}..{s['z_bot'].max():.0f} m  "
              f"литотипы: {', '.join(names)}")

    # Граничные условия
    x, ts = data.tsurf(0.0)
    x, qb = data.heat_flow(0.0)
    print(f'\n--- Граничные условия на 0 Ma ---')
    print(f'  Tsurf:  {ts.min():.1f} .. {ts.max():.1f} °C')
    print(f'  qbasal: {qb.min():.1f} .. {qb.max():.1f} мВт/м²')

    # Пример расчёта эффективных свойств
    print(f'\n--- Эффективные свойства (Layer_5, 75 Ma, x=500 m) ---')
    x, zt, zb = data.geometry('Layer_5', 75.0)
    idx = np.argmin(np.abs(x - 500))
    lith_code = data.lithology('Layer_5', 75.0)[idx]
    phi_pct = data.porosity('Layer_5', 75.0)[idx]
    phi = phi_pct / 100.0

    lt = data.get_lithotype(lith_code)
    T_K = 273.15 + 80  # допустим 80°C

    lam = effective_lambda(T_K, lt['lambda_20'], phi)
    rc = effective_rho_c(T_K, lt['c20_J'], lt['rho_s'], phi)
    A = effective_A(lt['As'], phi)

    print(f'  Литотип: {lt["name"]} (код {lith_code})')
    print(f'  φ = {phi_pct:.1f}%')
    print(f'  λ_eff = {lam:.3f} Вт/(м·К)')
    print(f'  (ρc)_eff = {rc:.0f} Дж/(м³·К)')
    print(f'  A_eff = {A:.4f} µВт/м³')
