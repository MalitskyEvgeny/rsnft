from pathlib import Path
import matplotlib.pyplot as plt
from basin_data import BasinData
from thermal_solver import BasinThermalSolver

if __name__ == '__main__':
    # 1. Укажите путь к папке с xlsx-файлами
    XLSX_DIR = Path(__file__).parent / "data1"

    # 2. Инициализация контейнера данных
    data = BasinData(XLSX_DIR)

    # 3. Запуск решателя прямой задачи (ALE + МКО)
    solver = BasinThermalSolver(data, nz_per_layer=5, tol=1e-3)
    results = solver.run()

    # 4. Визуализация результатов на современный момент (0 Ma)
    last_snap = results['snapshots'][-1]
    mesh = last_snap['mesh']
    T = last_snap['T']
    Ro = last_snap['Ro']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Карты температурного поля T [°C]
    c1 = ax1.pcolormesh(
        mesh.x_faces / 1000.0,
        mesh.z_nodes,         # Полная матрица узлов (nz + 1, nx + 1)
        T,                    # Матрица значений (nz, nx)
        cmap='inferno',
        shading='flat'
    )
    fig.colorbar(c1, ax=ax1, label='Температура [°C]')
    ax1.set_ylabel('Глубина [м]')
    ax1.set_title(f'Температурное поле на t = {last_snap["time"]} Ma')
    ax1.invert_yaxis()

    # Карты кинетической зрелости EASY%Ro [%]
    c2 = ax2.pcolormesh(
        mesh.x_faces / 1000.0,
        mesh.z_nodes,          # <-- Убрали срез [:, :-1], теперь здесь полная матрица (nz+1, nx+1)
        Ro,                    # Матрица значений зрелости (nz, nx)
        cmap='viridis',
        shading='flat',
        vmin=0.2,
        vmax=2.5
    )
    fig.colorbar(c2, ax=ax2, label='EASY %Ro [%]')
    ax2.set_xlabel('Профиль X [км]')
    ax2.set_ylabel('Глубина [м]')
    ax2.set_title(f'Отражательная способность витринита (%Ro) на t = {last_snap["time"]} Ma')
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig('basin_simulation_result.png', dpi=300, bbox_inches='tight')
    print("График успешно сохранен в файл 'basin_simulation_result.png'")
