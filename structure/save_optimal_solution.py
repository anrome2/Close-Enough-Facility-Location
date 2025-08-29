from datetime import datetime
import multiprocessing
import os
import pulp # Asegúrate de que pulp esté importado si lo usas en milp_ceflp o en create_params

# Importar las funciones necesarias
from algorithms.MILP import milp_ceflp
from main import create_params, setup_logger

manager = multiprocessing.Manager()
lock = manager.Lock()

# Define la función que será ejecutada por cada proceso
def milp_parallel(task_args):
    """
    Función wrapper para ejecutar milp_ceflp con argumentos desempaquetados.
    
    Args:
        task_args (tuple): Una tupla que contiene (path, i, result_dir, model_type, solver, time_limit).
    """
    path, instance, i, n_nodos, result_dir, problem_type, solver, time_limit, logger = task_args
    # Aquí puedes añadir cualquier inicialización específica del proceso si fuera necesaria
    # Por ejemplo, configurar loggers específicos para cada proceso si hay colisiones.
    print(f"Procesando instancia {i} de {problem_type} desde {path}...")
    with lock:
        milp_ceflp(params=create_params(path, i), 
                instance=instance, 
                n_nodos=n_nodos,
                result_dir=result_dir, 
                problem_type=problem_type, 
                optimizer=solver, 
                time_limit=time_limit,
                optimal_solution=True,
                logger=logger)
    print(f"Instancia {i} de {problem_type} finalizada.")


if __name__ == "__main__":
    # Directorio base donde están las instancias
    base_dir = "data/training_set"
    subfolders = ["n_10", "n_50", "n_100"]

    base_result_dir = "data/optimal_solutions/"
    os.makedirs(base_result_dir, exist_ok=True)
    log_filename = os.path.join(base_result_dir, f"execution.log")
    logger = setup_logger(log_filename)

    tasks = []
    idx = 0
    for sub in subfolders:
        folder_path = os.path.join(base_dir, sub)

        if not os.path.exists(folder_path):
            print(f"[WARNING] Carpeta {folder_path} no encontrada, se ignora.")
            continue

        for fname in sorted(os.listdir(folder_path)):
            if fname.endswith(".txt"):
                path = os.path.join(folder_path, fname)
                instance = os.path.splitext(fname)[0]
                # Agregar la tarea (ajusta si milp_parallel necesita más/menos parámetros)
                tasks.append((
                    path,                  # ruta al archivo
                    instance,            # nombre de instancia (p{j}_i{x})
                    idx,                   # índice único
                    sub,                   # identificador del problema (aquí usamos la carpeta, ej: n_10)
                    base_result_dir,       # carpeta de salida
                    "P2",                  # problema
                    "CPLEX",               # optimizador
                    1000,                   # algún parámetro (ej: límite de tiempo)
                    logger
                ))
                idx += 1

    num_processes = min(8, multiprocessing.cpu_count()) 
    print(f"Iniciando la ejecución paralela en {num_processes} procesos...")
    print(f"Ejecutando solución exacta para optimizador CPLEX...")
    print(f"Total de tareas a ejecutar: {len(tasks)}")

    chunk_size = 20  # número de tareas a ejecutar por lote
    for i in range(0, len(tasks), chunk_size):
        chunk = tasks[i:i+chunk_size]
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(milp_parallel, chunk)

    print(f"Todas las operaciones completadas. Revisa las carpetas '{base_result_dir}' para los logs y resultados.")