from datetime import datetime
import multiprocessing
import os
import pulp # Asegúrate de que pulp esté importado si lo usas en milp_ceflp o en create_params

# Importar las funciones necesarias
from algorithms.MILP import milp_ceflp
from main import create_params

# Define la función que será ejecutada por cada proceso
def milp_parallel(task_args):
    """
    Función wrapper para ejecutar milp_ceflp con argumentos desempaquetados.
    
    Args:
        task_args (tuple): Una tupla que contiene (path, i, result_dir, model_type, solver, time_limit).
    """
    path, i, instance_problem, result_dir, problem_type, solver, time_limit = task_args
    # Aquí puedes añadir cualquier inicialización específica del proceso si fuera necesaria
    # Por ejemplo, configurar loggers específicos para cada proceso si hay colisiones.
    print(f"Procesando instancia {i+1} de {problem_type} desde {path}...")
    milp_ceflp(params=create_params(path, i), 
               instance=i, 
               instance_problem=instance_problem,
               result_dir=result_dir, 
               problem_type=problem_type, 
               optimizer=solver, 
               time_limit=time_limit,
               optimal_solution=True)
    print(f"Instancia {i+1} de {problem_type} finalizada.")


if __name__ == "__main__":
    # Asegúrate de que este bloque solo se ejecute cuando el script es el proceso principal.
    # Esto es crucial para multiprocessing.

    tasks = []
    # Usar una variable para controlar el problema y evitar nombres de carpeta con "p0"
    for j_problem_index in range(0, 10): # Solo una iteración para "p1"
        problem_id = j_problem_index + 1
        instance_problem = f"p{problem_id}" # Esto hará que sea "p1"

        # La ruta del directorio de resultados debe ser un directorio, no un nombre de archivo .json
        # Ajusta esto si 'result_dir' en milp_ceflp espera un path a un archivo.
        base_result_dir = f"data/optimal_solutions/" # Usar una carpeta para cada tipo de problema
        os.makedirs(base_result_dir, exist_ok=True)
        
        for i_instance_index in range(0, 60): # Iteraciones para las 10 instancias (i1 a i10)
            instance_name = f"i{i_instance_index+1}"
            path = f"instances/{instance_problem}/{instance_name}.txt"
            
            # Cada elemento en 'tasks' debe ser una tupla de argumentos para 'milp_parallel'
            tasks.append((path, i_instance_index, instance_problem, base_result_dir, "P2", "CPLEX", 1000))

    num_processes = multiprocessing.cpu_count()
    print(f"Iniciando la ejecución paralela en {num_processes} procesos...")
    print(f"Ejecutando solución exacta para optimizador CPLEX...")

    # Imprimir las tareas para verificar que se generaron correctamente
    print(f"Total de tareas a ejecutar: {len(tasks)}")
    # print(f"Primeras 3 tareas: {tasks[:3]}") # Descomentar para depuración

    with multiprocessing.Pool(processes=num_processes) as pool:
        # pool.map espera una función que reciba un solo argumento iterable.
        # Por lo tanto, pasamos las tuplas de argumentos directamente a milp_parallel.
        results = pool.map(milp_parallel, tasks)

    print(f"Todas las operaciones completadas. Revisa las carpetas '{base_result_dir}' para los logs y resultados.")