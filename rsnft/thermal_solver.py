"""
thermal_solver.py
=================
Модуль решения прямой задачи теплопереноса и кинетической зрелости (EASY%Ro)
в двумерном осадочном чехле методом ALE + МКО (Finite Volume Method).
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from basin_data import (
    BasinData,
    effective_lambda,
    effective_rho_c,
    effective_A
)

# Перевод времени: 1 млн лет (Ma) в секунды
MA_TO_SEC = 1.0e6 * 365.25 * 86400.0


def project_x(arr_curr: np.ndarray, x_curr: np.ndarray, x_next: np.ndarray) -> np.ndarray:
    """
    Универсальная 1D-интерполяция физических полей по горизонтальной оси X.
    Корректно обрабатывает:
    - Разное количество точек
    - Разные координаты X и неравномерный шаг сетки
    - 2D (nz, nx) и 3D (nz, nx, n_components) массивы
    """
    x_c = np.asarray(x_curr, dtype=np.float64).ravel()
    x_n = np.asarray(x_next, dtype=np.float64).ravel()

    if len(x_c) == len(x_n) and np.allclose(x_c, x_n):
        return arr_curr.copy()

    sort_idx = np.argsort(x_c)
    x_c_sorted = x_c[sort_idx]

    if arr_curr.ndim == 2:
        nz_curr, _ = arr_curr.shape
        nx_next = len(x_n)
        arr_interp = np.zeros((nz_curr, nx_next), dtype=arr_curr.dtype)
        for r in range(nz_curr):
            y_sorted = arr_curr[r, sort_idx]
            arr_interp[r, :] = np.interp(x_n, x_c_sorted, y_sorted)
        return arr_interp

    elif arr_curr.ndim == 3:
        nz_curr, _, n_comp = arr_curr.shape
        nx_next = len(x_n)
        arr_interp = np.zeros((nz_curr, nx_next, n_comp), dtype=arr_curr.dtype)
        for r in range(nz_curr):
            for k in range(n_comp):
                y_sorted = arr_curr[r, :, k][sort_idx]
                arr_interp[r, :, k] = np.interp(x_n, x_c_sorted, y_sorted)
        return arr_interp
    else:
        raise ValueError(f"Неподдерживаемая размерность массива: {arr_curr.ndim}")


class ALEMesh2D:
    """
    Класс двумерной структурированной сетки МКО для осадочного чехла.
    Поддерживает ALE-перестроение при уплотнении и осадконакоплении.
    """

    def __init__(self, snapshot_data, lithotypes_dict, nz_per_layer=5):
        self.nz_per_layer = nz_per_layer
        self.lithotypes = lithotypes_dict
        self.build_mesh(snapshot_data)

    def build_mesh(self, snapshot_data):
        """Строит 2D сетку конечных объёмов на заданный момент времени."""
        self.n_layers = len(snapshot_data)
        self.x = np.asarray(snapshot_data[0]['x'], dtype=np.float64).squeeze()
        self.nx = len(self.x) - 1  # число столбцов ячеек (по x)
        self.nz = self.n_layers * self.nz_per_layer  # общее число ячеек по z

        self.x_faces = np.array(self.x, dtype=np.float64).copy()
        self.x_centers = 0.5 * (self.x_faces[:-1] + self.x_faces[1:])
        self.dx = np.diff(self.x_faces)  # (nx,)

        self.z_nodes = np.zeros((self.nz + 1, self.nx + 1))
        self.layer_names = []

        self.cell_lith = np.zeros((self.nz, self.nx), dtype=int)
        self.cell_poro = np.zeros((self.nz, self.nx), dtype=np.float64)

        def _fit_to_nx(arr, target_len, fill_dtype=float):
            arr_1d = np.asarray(arr, dtype=fill_dtype).squeeze()
            if arr_1d.ndim == 0:
                return np.full(target_len, arr_1d.item(), dtype=fill_dtype)
            if len(arr_1d) >= target_len:
                return arr_1d[:target_len]
            return np.pad(arr_1d, (0, target_len - len(arr_1d)), mode='edge')

        for l_idx, layer in enumerate(snapshot_data):
            self.layer_names.append(layer['layer'])

            z_top = np.asarray(layer['z_top'], dtype=np.float64).squeeze()
            z_bot = np.asarray(layer['z_bot'], dtype=np.float64).squeeze()

            lith_cells = _fit_to_nx(layer['lith'], self.nx, fill_dtype=int)
            poro_cells = _fit_to_nx(layer['porosity'], self.nx, fill_dtype=float) / 100.0

            def _interp_z(z_arr):
                if z_arr.ndim == 0:
                    return np.full_like(self.x_faces, float(z_arr))
                if len(z_arr) == self.nx:
                    return np.interp(self.x_faces, self.x_centers, z_arr)
                elif len(z_arr) == len(self.x_faces):
                    return np.interp(self.x_faces, self.x_faces, z_arr)
                else:
                    xp_src = np.linspace(self.x_faces[0], self.x_faces[-1], len(z_arr))
                    return np.interp(self.x_faces, xp_src, z_arr)

            z_top_faces = _interp_z(z_top)
            z_bot_faces = _interp_z(z_bot)

            r_start = l_idx * self.nz_per_layer
            r_end = (l_idx + 1) * self.nz_per_layer

            for k in range(self.nz_per_layer):
                r = r_start + k
                frac_top = k / self.nz_per_layer
                self.z_nodes[r, :] = (1.0 - frac_top) * z_top_faces + frac_top * z_bot_faces

                self.cell_lith[r, :] = lith_cells
                self.cell_poro[r, :] = poro_cells

            if l_idx == self.n_layers - 1:
                self.z_nodes[r_end, :] = z_bot_faces

        self.update_geometry()

    def update_geometry(self):
        """Перерасчёт объёмов ячеек и площадей граней."""
        self.dz_nodes = np.diff(self.z_nodes, axis=0)
        self.dz_cell = 0.5 * (self.dz_nodes[:, :-1] + self.dz_nodes[:, 1:])

        self.z_centers = 0.5 * (self.z_nodes[:-1, :] + self.z_nodes[1:, :])
        self.z_cell_centers = 0.5 * (self.z_centers[:, :-1] + self.z_centers[:, 1:])

        self.volume = np.outer(np.ones(self.nz), self.dx) * self.dz_cell


class EasyRoKinetics:
    """Модуль кинетического расчёта отражательной способности витринита EASY%Ro."""

    def __init__(self, easy_ro_dict):
        self.E_J = easy_ro_dict['E_J']        # [Дж/моль]
        self.f = easy_ro_dict['f']            # весовые доли
        self.A = easy_ro_dict['A']            # [1/с]
        self.Rg = easy_ro_dict['Rg_J']        # 8.314 Дж/(моль·К)

    def step(self, x_prev, T_degC, dt_sec):
        if dt_sec == 0.0:
            W = np.sum(self.f * (1.0 - x_prev), axis=-1)
            Ro = np.exp(-1.6 + 3.7 * W)
            return x_prev.copy(), Ro

        T_K = T_degC + 273.15
        T_K_flat = T_K.reshape(-1, 1)

        k = self.A * np.exp(-self.E_J / (self.Rg * T_K_flat))

        x_prev_flat = x_prev.reshape(-1, 20)
        x_next_flat = x_prev_flat * np.exp(-k * dt_sec)
        x_next = x_next_flat.reshape(x_prev.shape)

        W = np.sum(self.f * (1.0 - x_next), axis=-1)
        Ro = np.exp(-1.6 + 3.7 * W)

        return x_next, Ro


class BasinThermalSolver:
    """
    Основной класс-решатель прямой задачи теплопереноса и кинетической зрелости.
    """

    def __init__(self, basin_data: BasinData, nz_per_layer=5, tol=1.0e-3, max_iter=15):
        self.data = basin_data
        self.nz_per_layer = nz_per_layer
        self.tol = tol
        self.max_iter = max_iter
        self.ro_kinetics = EasyRoKinetics(basin_data.easy_ro)

    def _get_cell_properties(self, mesh: ALEMesh2D, T_degC: np.ndarray):
        """Расчёт эффективных теплофизических свойств для каждой ячейки."""
        T_K = T_degC + 273.15
        nz, nx = mesh.nz, mesh.nx

        lam_eff = np.zeros((nz, nx))
        rhoc_eff = np.zeros((nz, nx))
        A_eff = np.zeros((nz, nx))

        for r in range(nz):
            for c in range(nx):
                code = mesh.cell_lith[r, c]
                phi = mesh.cell_poro[r, c]
                lt = self.data.get_lithotype(code)

                lam_eff[r, c] = effective_lambda(T_K[r, c], lt['lambda_20'], phi)
                rhoc_eff[r, c] = effective_rho_c(T_K[r, c], lt['c20_J'], lt['rho_s'], phi)
                A_eff[r, c] = effective_A(lt['As'], phi) * 1.0e-6

        return lam_eff, rhoc_eff, A_eff

    def _assemble_solve_fvm(self, mesh_old: ALEMesh2D, mesh_new: ALEMesh2D,
                            T_old_raw: np.ndarray, T_old: np.ndarray, T_guess: np.ndarray,
                            dt_sec: float, t_new: float):
        """Сборка СЛАУ МКО в ALE-формулировке и её решение."""
        nz, nx = mesh_new.nz, mesh_new.nx
        N = nz * nx

        # Граничные условия
        x_bc_raw, tsurf_bc_raw = self.data.tsurf(t_new)
        x_bc_q, qbasal_bc_raw = self.data.heat_flow(t_new)

        # Безопасная интерполяция Tsurf
        mask_t = ~np.isnan(x_bc_raw) & ~np.isnan(tsurf_bc_raw)
        x_bc_t = x_bc_raw[mask_t]
        tsurf_bc = tsurf_bc_raw[mask_t]
        sort_t = np.argsort(x_bc_t)
        tsurf = np.interp(mesh_new.x_centers, x_bc_t[sort_t], tsurf_bc[sort_t])

        # Безопасная интерполяция Qbasal
        mask_q = ~np.isnan(x_bc_q) & ~np.isnan(qbasal_bc_raw)
        x_bc_q_clean = x_bc_q[mask_q]
        qbasal_bc = qbasal_bc_raw[mask_q]
        sort_q = np.argsort(x_bc_q_clean)
        qbasal = np.interp(mesh_new.x_centers, x_bc_q_clean[sort_q], qbasal_bc[sort_q]) * 1.0e-3

        # Свойства среды на новой сетке
        lam, rhoc_new, A_new = self._get_cell_properties(mesh_new, T_guess)

        # Теплоёмкости на старой сетке
        _, rhoc_old_raw, _ = self._get_cell_properties(mesh_old, T_old_raw)

        # Проекция объёмов и теплоёмкостей с mesh_old на mesh_new по горизонтали X
        vol_old_x = project_x(mesh_old.volume, mesh_old.x_centers, mesh_new.x_centers)
        rhoc_old_x = project_x(rhoc_old_raw, mesh_old.x_centers, mesh_new.x_centers)

        dz_diff = mesh_new.nz - mesh_old.nz

        vol_old_proj = np.zeros((nz, nx))
        rhoc_old_proj = np.zeros((nz, nx))

        if dz_diff > 0:
            vol_old_proj[dz_diff:, :] = vol_old_x
            rhoc_old_proj[dz_diff:, :] = rhoc_old_x
            vol_old_proj[:dz_diff, :] = 0.0
            rhoc_old_proj[:dz_diff, :] = rhoc_new[:dz_diff, :]
        elif dz_diff < 0:
            k = mesh_old.nz - mesh_new.nz
            vol_old_proj = vol_old_x[k:, :].copy()
            rhoc_old_proj = rhoc_old_x[k:, :].copy()
        else:
            vol_old_proj = vol_old_x.copy()
            rhoc_old_proj = rhoc_old_x.copy()

        # Быстрая сборка СЛАУ с использованием списков COO-формата
        rows = []
        cols = []
        data = []
        b_vec = np.zeros(N)

        def idx(r, c):
            return r * nx + c

        z_top_face_centers = 0.5 * (mesh_new.z_nodes[:-1, :-1] + mesh_new.z_nodes[:-1, 1:])

        for r in range(nz):
            for c in range(nx):
                ij = idx(r, c)

                vol_new = mesh_new.volume[r, c]
                vol_old = vol_old_proj[r, c]
                dx_c = mesh_new.dx[c]

                # 1. Аккумуляционный член ALE
                a_P = (rhoc_new[r, c] * vol_new) / dt_sec
                rhs = (rhoc_old_proj[r, c] * vol_old * T_old[r, c]) / dt_sec + A_new[r, c] * vol_new

                # 2. Потоки через верхнюю грань (z_top)
                if r == 0:
                    dz_top = mesh_new.z_cell_centers[r, c] - z_top_face_centers[r, c]
                    trans_top = (lam[r, c] * dx_c) / dz_top
                    a_P += trans_top
                    rhs += trans_top * tsurf[c]
                else:
                    dz_top = mesh_new.z_cell_centers[r, c] - mesh_new.z_cell_centers[r - 1, c]
                    lam_top = 2.0 * lam[r, c] * lam[r - 1, c] / (lam[r, c] + lam[r - 1, c] + 1e-12)
                    trans_top = (lam_top * dx_c) / dz_top
                    a_P += trans_top
                    rows.append(ij)
                    cols.append(idx(r - 1, c))
                    data.append(-trans_top)

                # 3. Потоки через нижнюю грань (z_bot)
                if r == nz - 1:
                    rhs += qbasal[c] * dx_c
                else:
                    dz_bot = mesh_new.z_cell_centers[r + 1, c] - mesh_new.z_cell_centers[r, c]
                    lam_bot = 2.0 * lam[r, c] * lam[r + 1, c] / (lam[r, c] + lam[r + 1, c] + 1e-12)
                    trans_bot = (lam_bot * dx_c) / dz_bot
                    a_P += trans_bot
                    rows.append(ij)
                    cols.append(idx(r + 1, c))
                    data.append(-trans_bot)

                # 4. Потоки через левую грань (x_left)
                if c > 0:
                    dz_left = mesh_new.dz_cell[r, c]
                    dx_left = mesh_new.x_centers[c] - mesh_new.x_centers[c - 1]
                    lam_left = 2.0 * lam[r, c] * lam[r, c - 1] / (lam[r, c] + lam[r, c - 1] + 1e-12)
                    trans_left = (lam_left * dz_left) / dx_left
                    a_P += trans_left
                    rows.append(ij)
                    cols.append(idx(r, c - 1))
                    data.append(-trans_left)

                # 5. Потоки через правую грань (x_right)
                if c < nx - 1:
                    dz_right = mesh_new.dz_cell[r, c]
                    dx_right = mesh_new.x_centers[c + 1] - mesh_new.x_centers[c]
                    lam_right = 2.0 * lam[r, c] * lam[r, c + 1] / (lam[r, c] + lam[r, c + 1] + 1e-12)
                    trans_right = (lam_right * dz_right) / dx_right
                    a_P += trans_right
                    rows.append(ij)
                    cols.append(idx(r, c + 1))
                    data.append(-trans_right)

                # Главная диагональ
                rows.append(ij)
                cols.append(ij)
                data.append(a_P)

                b_vec[ij] = rhs

        A_mat = csr_matrix((data, (rows, cols)), shape=(N, N))
        T_vec = spsolve(A_mat, b_vec)
        return T_vec.reshape((nz, nx))

    def run(self):
        """Запуск полного цикла моделирования."""
        raw_times = list(self.data.times)
        if raw_times[0] < raw_times[-1]:
            times = raw_times[::-1]
        else:
            times = raw_times

        n_steps = len(times)

        results = {
            'times': times,
            'snapshots': []
        }

        # --- 1. Начальное состояние (самая древняя эпоха, t_max) ---
        t_curr = times[0]
        snap_curr = self.data.snapshot(t_curr)
        mesh_curr = ALEMesh2D(snap_curr, self.data.lithotypes, self.nz_per_layer)

        x_bc, tsurf = self.data.tsurf(t_curr)
        T_curr = np.zeros((mesh_curr.nz, mesh_curr.nx))
        for c in range(mesh_curr.nx):
            ts = np.interp(mesh_curr.x_centers[c], x_bc, tsurf)
            T_curr[:, c] = ts + 0.03 * mesh_curr.z_cell_centers[:, c]

        x_ro_curr = np.ones((mesh_curr.nz, mesh_curr.nx, 20), dtype=np.float64)
        _, Ro_curr = self.ro_kinetics.step(x_ro_curr, T_curr, dt_sec=0.0)

        results['snapshots'].append({
            'time': t_curr,
            'mesh': mesh_curr,
            'T': T_curr.copy(),
            'Ro': Ro_curr.copy()
        })

        print(f"[ALE-FVM] Шаг 1/{n_steps}: t = {t_curr:g} Ma | "
              f"T_min = {T_curr.min():.1f}°C, T_max = {T_curr.max():.1f}°C")

        # --- 2. Эволюция во времени (к настоящему моменту t = 0) ---
        for step in range(1, n_steps):
            t_next = times[step]
            dt_Ma = abs(t_next - t_curr)
            dt_sec = dt_Ma * MA_TO_SEC

            snap_next = self.data.snapshot(t_next)
            mesh_next = ALEMesh2D(snap_next, self.data.lithotypes, self.nz_per_layer)

            # Перепроекция по горизонтали (X) с mesh_curr на mesh_next
            T_curr_x = project_x(T_curr, mesh_curr.x_centers, mesh_next.x_centers)
            x_ro_curr_x = project_x(x_ro_curr, mesh_curr.x_centers, mesh_next.x_centers)

            dz_diff = mesh_next.nz - mesh_curr.nz

            if dz_diff == 0:
                T_old_projected = T_curr_x.copy()
                x_ro_projected = x_ro_curr_x.copy()
            elif dz_diff > 0:
                # Осадконакопление (добавление новых слоёв сверху)
                T_old_projected = np.zeros((mesh_next.nz, mesh_next.nx))
                x_ro_projected = np.ones((mesh_next.nz, mesh_next.nx, 20))

                T_old_projected[dz_diff:, :] = T_curr_x
                x_ro_projected[dz_diff:, :, :] = x_ro_curr_x

                x_bc_n, ts_n = self.data.tsurf(t_next)
                for c in range(mesh_next.nx):
                    ts_val = np.interp(mesh_next.x_centers[c], x_bc_n, ts_n)
                    T_old_projected[:dz_diff, c] = ts_val
            else:
                # На случай геологической эрозии (размыва слоёв)
                k = mesh_curr.nz - mesh_next.nz
                T_old_projected = T_curr_x[k:, :].copy()
                x_ro_projected = x_ro_curr_x[k:, :, :].copy()

            # Итерации Пикара для учёта нелинейной теплопроводности
            T_guess = T_old_projected.copy()
            for it in range(self.max_iter):
                T_new = self._assemble_solve_fvm(
                    mesh_old=mesh_curr,
                    mesh_new=mesh_next,
                    T_old_raw=T_curr,
                    T_old=T_old_projected,
                    T_guess=T_guess,
                    dt_sec=dt_sec,
                    t_new=t_next
                )

                err = np.max(np.abs(T_new - T_guess))
                T_guess = T_new.copy()
                if err < self.tol:
                    break

            x_ro_next, Ro_next = self.ro_kinetics.step(x_ro_projected, T_new, dt_sec)

            t_curr = t_next
            mesh_curr = mesh_next
            T_curr = T_new
            x_ro_curr = x_ro_next
            Ro_curr = Ro_next

            results['snapshots'].append({
                'time': t_curr,
                'mesh': mesh_curr,
                'T': T_curr.copy(),
                'Ro': Ro_curr.copy()
            })

            print(f"[ALE-FVM] Шаг {step+1}/{n_steps}: t = {t_curr:g} Ma | "
                  f"T_min = {T_curr.min():.1f}°C, T_max = {T_curr.max():.1f}°C, "
                  f"Ro_max = {Ro_next.max():.2f}%")

        return results
