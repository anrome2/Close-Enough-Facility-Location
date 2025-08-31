import json
import logging
import multiprocessing
import os
from datetime import datetime
import time

from Comittee_agent import run_genetic_committee, run_genetic_once, run_grasp_committee, run_grasp_hyperparam_search, run_grasp_once, run_tabu_committee, run_tabu_hyperparam_search, run_tabu_once
from Orchestrator import generate_final_report, generate_genetic_param_combinations, generate_grasp_param_combinations, generate_tabu_param_combinations, run_committee_multiple_instances, run_hyperparameter_search_multiple_instances, run_single_genetic_execution_improved, run_single_grasp_execution_improved, run_single_tabu_execution_improved
from algorithms.MILP import milp_ceflp as MILP
from structure.create_instances import create_params

# C:\Users\Andrea\Documents\TFM\.venv\Scripts\Activate.ps1
# source venv_ceflp/bin/activate

CONFIG = {
    'testing': True,
    'n_nodos': ["n_100"], # Un listado, las opciones son ["n_10", "n_50", "n_100"]
    'save_results': True,
    'problem': 'P2', # Puede ser 'P1' o 'P2'
    'algorithm': 'GENETIC',  # Puede ser 'GRASP' o 'MILP' o 'TABU' o 'GENETIC'
    'optimization_solver': 'CPLEX',  # Puede ser 'CBC', 'GLPK' o 'CPLEX'
    'inicialization': 'random',  # Puede ser 'random', 'kmeans' o 'greedy'
    'mode': 'hyperparameters', # Puede ser 'comitee', 'hyperparam' o 'single'
    'type_search': 'random',  # Tipo de búsqueda de hiperparámetros, puede ser 'grid', 'random' o 'bayesian'
    'alpha': 0.65,  # Parámetro de aleatoriedad para GREEDY
    'frac_neighbors': 3,  # Fracción a dividir el total de clientes para obtener el número de vecinos a generar por iteración
    'tabu_tenure': 0.35,  # Tenencia para el algoritmo Tabu
    'tournament': 2,
    'gamma_f': 0.5,
    'gamma_q': 0.5,
    'time_limit': 50,  # Límite de tiempo en segundos para la resolución del MILP
    'max_iter': 50,  # Máximo número de iteraciones para el algoritmo Tabu
    'num_runs': 2,
    'num_processes': 16
}

def setup_logger(log_file_path):
    """
    configura un logger para escribir mensajes en un archivo y en la consola.
    """
    logger = logging.getLogger('CEFLP_Logger')
    logger.setLevel(logging.INFO) # Establece el nivel mínimo de mensajes a registrar

    # Crea un handler para escribir en un archivo
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO) # Nivel de mensajes para el archivo

    # Crea un handler para imprimir en la consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) # Nivel de mensajes para la consola

    # Define el formato de los mensajes
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Añade los handlers al logger
    # Evita añadir múltiples handlers si ya existen (puede pasar en bucles)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

def load_instances(subfolders, testing: bool = False)->list:
    """Carga todas las instancias de testing desde data/testing_sets"""
    if testing:
        instances_dir = "data/testing_set"
    else:
        instances_dir = "data/training_set"
    instances = []
    
    if not os.path.exists(instances_dir):
        print(f"WARNING: No se encuentra el directorio {instances_dir}")
        return instances
    
    # Buscar archivos de instancias (asumiendo que son .json o .txt)
    for sub in subfolders:
        folder_path = os.path.join(instances_dir, sub)

        if not os.path.exists(folder_path):
            print(f"[WARNING] Carpeta {folder_path} no encontrada, se ignora.")
            continue

        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder_path, filename)
                # Extraer información del nombre del archivo
                # Asumiendo formato como p1_i1.json, p2_i15.json, etc.
                instance_name = filename.split('.')[0]  # Quitar extensión
                instances.append((instance_name, filepath))
    
    return instances

# def get_instance_paths(config):
#     """Devuelve una lista de rutas de instancias en base a config."""
#     base_dir = "data/testing_set" if config["testing"] else "data/training_set"
#     instance_paths = []

#     for n in config["n_nodos"]:
#         folder = os.path.join(base_dir, f"n_{n}")
#         if not os.path.exists(folder):
#             print(f"[WARNING] Carpeta {folder} no encontrada, se ignora.")
#             continue
#         files = sorted(f for f in os.listdir(folder) if f.endswith(".txt"))
#         for idx, fname in enumerate(files):
#             instance_paths.append((idx, os.path.join(folder, fname), config))
#     return instance_paths
#
# def executeInstance(
#                 path, 
#                 instance: int, 
#                 logger: logging, 
#                 result_dir, 
#                 alpha: float = 0.3, 
#                 frac_neighbors: int = 4, 
#                 problem: str = 'P1', 
#                 algorithm: str = "GRASP", 
#                 optimizer: str = "CBC", 
#                 inicialization: str = "random", 
#                 tabu_tenure: float = 0.25,
#                 comitee: bool = False, 
#                 hiperparam_search: bool = False,
#                 type_search: str = 'grid',  
#                 num_runs: int = 4,
#                 time_limit = None,
#                 max_iter: int = 10
#             ):
#     start_time = time.time()
#     params = create_params(path, instance)
#     logger.info(f"Tiempo en crear parámetros {time.time()-start_time}")
#     if algorithm == "GRASP":
#         if hiperparam_search:
#             run_grasp_hyperparam_search(params=params, instance=instance, result_dir=result_dir, logger=logger, problem=problem, num_runs=num_runs, type_search=type_search)
#         elif comitee:
#             run_grasp_committee(params = params, instance=instance, alpha=alpha, frac_neighbors=frac_neighbors, result_dir=result_dir, problem=problem, max_iter=max_iter, logger=logger, num_runs=num_runs)
#         else:
#             run_grasp_once(params = params, instance=instance, alpha=alpha, frac_neighbors=frac_neighbors, result_dir=result_dir, problem=problem, max_iter=max_iter, logger=logger)
        
#     elif algorithm == "MILP":
#         MILP(problem_type=problem, optimizer=optimizer, params = params, instance=instance, instance_problem="p1", result_dir=result_dir, time_limit=time_limit, logger=logger)

#     elif algorithm == "TABU":
#         if hiperparam_search:
#             run_tabu_hyperparam_search(params=params, instance=instance, result_dir=result_dir, problem=problem, logger=logger, time_limit=time_limit, num_runs=num_runs, type_search=type_search)
#         elif comitee:
#             run_tabu_committee(params=params, instance=instance, tabu_tenure=tabu_tenure, result_dir=result_dir,  problem=problem,logger=logger, inicialization=inicialization, time_limit=time_limit, num_runs=num_runs)
#         else:
#             run_tabu_once(params=params, instance=instance, tabu_tenure=tabu_tenure, result_dir=result_dir,  problem=problem,logger=logger, inicialization=inicialization, time_limit=time_limit)

#     elif algorithm == "GENETIC":
#         if comitee:
#             run_genetic_committee(params=params, instance=instance, result_dir=result_dir,  problem=problem, inicializacion=inicialization, logger=logger)
#         else:
#             run_genetic_once(params=params, instance=instance, result_dir=result_dir,  problem=problem, inicializacion=inicialization, logger=logger)

# --- Función adaptada para la paralelización ---
# def run_single_instance(args_tuple):

#     i, global_timestamp, current_config = args_tuple

#     # Extraer los parámetros de configuración comunes del 'current_config'
#     algorithm = current_config['algorithm']
#     optimizer = current_config['optimization_solver']
#     problem = current_config['problem']
#     type_search = current_config.get('type_search', 'grid')
#     tabu_tenure = current_config.get('tabu_tenure', 0.25)
#     inicialization = current_config['inicialization']
#     alpha = current_config.get('alpha', 0.3)
#     frac_neighbors = current_config.get('frac_neighbors', 4)
#     time_limit = current_config['time_limit']
#     num_runs = current_config.get('num_runs', 4)
#     max_iter = current_config['max_iter']
#     hiperparam_search = current_config.get('hyperparam_search', False)
#     comitee = current_config.get('comitee', False)

#     instance_name = f"i{i+1}"
#     path = f"instances/p1/{instance_name}.txt"

#     # Determinar el directorio de resultados
#     if algorithm == "MILP":
#         result_dir = f"output/{problem}/{algorithm}/{optimizer}/{global_timestamp}"
#     elif algorithm == "GRASP":
#         if hiperparam_search:
#             result_dir = f"output/{problem}/{algorithm}/hyperparam_search/{type_search}/{global_timestamp}"
#         elif comitee:
#             result_dir = f"output/{problem}/{algorithm}/comitee/{global_timestamp}"
#         else:
#             result_dir = f"output/{problem}/{algorithm}/once/{global_timestamp}"
#     elif algorithm == "TABU":
#         if hiperparam_search:
#             result_dir = f"output/{problem}/{algorithm}/hyperparam_search/{type_search}/{global_timestamp}"
#         elif comitee:
#             result_dir = f"output/{problem}/{algorithm}/comitee/{inicialization}/{global_timestamp}"
#         else:
#             result_dir = f"output/{problem}/{algorithm}/once/{inicialization}/{global_timestamp}"
#     elif algorithm == "GENETIC":
#             if comitee:
#                 result_dir = f"output/{problem}/{algorithm}/comitee/{inicialization}/{global_timestamp}"
#             else:
#                 result_dir = f"output/{problem}/{algorithm}/{inicialization}/{global_timestamp}"

#     os.makedirs(result_dir, exist_ok=True)
#     log_filename = os.path.join(result_dir, f"execution_{instance_name}_{global_timestamp}.log")
#     instance_logger = setup_logger(log_filename) # Cada instancia, su propio logger

#     instance_logger.info(f"--- Inicio de la ejecución para {instance_name}: {global_timestamp} ---")
#     print(f"  [Proceso {os.getpid()}] Usando el solver: {optimizer}" if algorithm == "MILP" else f"  [Proceso {os.getpid()}] Usando {algorithm}")

#     try:
#         executeInstance(
#             path=path,
#             problem=problem,
#             algorithm=algorithm,
#             optimizer=optimizer,
#             inicialization=inicialization,
#             alpha=alpha,
#             frac_neighbors=frac_neighbors,
#             tabu_tenure=tabu_tenure,
#             instance=i,
#             comitee=comitee,
#             hiperparam_search=hiperparam_search,
#             type_search=type_search,
#             result_dir=result_dir,
#             time_limit=time_limit,
#             num_runs=num_runs,
#             max_iter=max_iter,
#             logger=instance_logger,
#         )
#         instance_logger.info(f"--- Fin de la ejecución para {instance_name} ---")
#         return f"Instancia {instance_name} completada con éxito."
#     except Exception as e:
#         instance_logger.error(f"Error procesando instancia {instance_name}: {e}")
#         return f"Error en instancia {instance_name}: {e}"
    
def main(config: dict):
    global_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    testing = config.get('testing', False)
    n_nodos = config.get('n_nodos', ['n_10', 'n_50', 'n_100'])
    mode = config.get('mode', 'single')
    algorithm = config.get('algorithm', 'tabu')
    problem = config.get('problem', 'P2')
    num_runs = config.get('num_runs', 4)
    num_processes = config.get('num_processes', None)
    result_dir = f"output/{problem}/{algorithm}"

    # Obtener las instancias según sea test o train
    instances = load_instances(subfolders=n_nodos, testing=testing)

    if mode == 'committee':
        result_base_dir = f"{result_dir}/comittee/{global_timestamp}"
        # Ejecutar comités con parámetros por defecto
        if algorithm == 'GRASP':
            # Parámetros por defecto para GRASP
            algorithm_params = []
            alpha = config.get('alpha', 0.3)
            frac_neighbors = config.get('frac_neighbors', 4)
            max_iter = config.get('max_iter', 20)
            for i, instance in enumerate(instances):
                instance_name = instance[0]
                path = instance[1]
                params = {
                    'algorithm_type': 'GRASP',
                    'params': create_params(instance=i, path=path),
                    'alpha': alpha,
                    'frac_neighbors': frac_neighbors,
                    'problem': problem,
                    'max_iter': max_iter
                }
                algorithm_params.append(params)
            
        elif algorithm == 'GENETIC':
            # Parámetros por defecto para Genetic
            algorithm_params = []
            inicialization = config.get('inicialization', 'random')
            generations = config.get('max_iter', 20)
            tournament = config.get('tournament', 5)
            for i, instance in enumerate(instances):
                instance_name = instance[0]
                path = instance[1]
                params = {
                    'algorithm_type': 'GENETIC',
                    'params': create_params(instance=i, path=path),
                    'generations': generations,
                    'mutation_rate': 0.05,
                    'crossover_rate': 0.95,
                    'tournament': tournament,
                    'inicializacion': inicialization,
                    'problem': problem
                }
                algorithm_params.append(params)
            
        elif algorithm == 'TABU':
            # Parámetros por defecto para Tabu
            algorithm_params = []
            max_iter = config.get('max_iter', 20)
            tabu_tenure = config.get('tabu_tenure', 0.25)
            inicialization = config.get('inicialization', 'random')
            time_limit = config.get('time_limit')
            gamma_f = config.get('gamma_f', 0.5)
            gamma_q = config.get('gamma_q', 0.5)
            for i, instance in enumerate(instances):
                instance_name = instance[0]
                path = instance[1]
                params = {
                    'algorithm_type': 'TABU',
                    'params': create_params(instance=i, path=path),
                    'tabu_tenure': tabu_tenure,
                    'inicializacion': inicialization,
                    'max_iter': max_iter,
                    'gamma_f': gamma_f,
                    'gamma_q': gamma_q,
                    'time_limit': time_limit,
                    'problem': problem
                }
                algorithm_params.append(params)
        
        # Ejecutar comités
        results = run_committee_multiple_instances(
            algorithm_params, instances, n_nodos[0], result_base_dir,
            algorithm, num_runs, num_processes
        )
        
    elif mode == 'hyperparameters':
        search_type = config.get('type_search', 'grid')
        result_base_dir = f"{result_dir}/hyperparameters/{search_type}/{global_timestamp}"
        # Búsqueda de hiperparámetros
        if algorithm == 'GRASP':
            param_combinations = generate_grasp_param_combinations(search_type)
        elif algorithm == 'GENETIC':
            param_combinations = generate_genetic_param_combinations(search_type)
        elif algorithm == 'TABU':
            param_combinations = generate_tabu_param_combinations(search_type)
        else:
            raise ValueError(f"Algoritmo desconocido: {algorithm}")
        
        # Ejecutar búsqueda de hiperparámetros
        results = run_hyperparameter_search_multiple_instances(
            algorithm, param_combinations, instances, n_nodos[0],
            result_base_dir, num_runs, num_processes
        )
    
    else:
        result_base_dir = f"{result_dir}/single/{global_timestamp}"
        if algorithm == 'GRASP':
            # Parámetros por defecto para GRASP
            algorithm_params = []
            alpha = config.get('alpha', 0.3)
            frac_neighbors = config.get('frac_neighbors', 4)
            max_iter = config.get('max_iter', 20)
            for i, instance in enumerate(instances):
                instance_name = instance[0]
                path = instance[1]
                params = {
                    'algorithm_type': 'GRASP',
                    'params': create_params(instance=i, path=path),
                    'instance': i,
                    'instance_name': instance_name,
                    'alpha': alpha,
                    'frac_neighbors': frac_neighbors,
                    'problem': problem,
                    'max_iter': max_iter,
                    'result_dir': result_base_dir,
                    'run_idx': i,
                    'temp_dir': os.path.join('/tmp', f'{algorithm}_single')
                }
                algorithm_params.append(params)

            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(run_single_grasp_execution_improved, algorithm_params)
            
        elif algorithm == 'GENETIC':
            # Parámetros por defecto para Genetic
            algorithm_params = []
            inicialization = config.get('inicialization', 'random')
            generations = config.get('max_iter', 20)
            tournament = config.get('tournament', 5)
            for i, instance in enumerate(instances):
                instance_name = instance[0]
                path = instance[1]
                params = {
                    'algorithm_type': 'GENETIC',
                    'params': create_params(instance=i, path=path),
                    'instance': i,
                    'instance_name': instance_name,
                    'generations': generations,
                    'mutation_rate': 0.05,
                    'crossover_rate': 0.95,
                    'tournament': tournament,
                    'inicializacion': inicialization,
                    'problem': problem,
                    'run_idx': i,
                    'temp_dir': os.path.join('/tmp', f'{algorithm}_single')
                }
                algorithm_params.append(params)

            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(run_single_genetic_execution_improved, algorithm_params)
            
        elif algorithm == 'TABU':
            # Parámetros por defecto para Tabu
            algorithm_params = []
            max_iter = config.get('max_iter', 20)
            tabu_tenure = config.get('tabu_tenure', 0.25)
            inicialization = config.get('inicialization', 'random')
            time_limit = config.get('time_limit')
            gamma_f = config.get('gamma_f', 0.5)
            gamma_q = config.get('gamma_q', 0.5)
            for i, instance in enumerate(instances):
                instance_name = instance[0]
                path = instance[1]
                params = {
                    'algorithm_type': 'TABU',
                    'params': create_params(instance=i, path=path),
                    'instance': i,
                    'instance_name': instance_name,
                    'tabu_tenure': tabu_tenure,
                    'inicializacion': inicialization,
                    'max_iter': max_iter,
                    'gamma_f': gamma_f,
                    'gamma_q': gamma_q,
                    'time_limit': time_limit,
                    'problem': problem,
                    'run_idx': i,
                    'temp_dir': os.path.join('/tmp', f'{algorithm}_single')
                }
                algorithm_params.append(params)

            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(run_single_tabu_execution_improved, algorithm_params)
        else:
            raise ValueError(f"Algoritmo desconocido: {algorithm}")
    
    # Generar reporte final
    generate_final_report(results, result_base_dir, algorithm, mode)
    
    print(f"Experimento completado. Resultados en: {result_base_dir}")
    return results


# --- Bloque principal para la ejecución paralela ---
if __name__ == "__main__":
    main(CONFIG)
    # if hiperparam_search or comitee:
    #     # Ejecución secuencial
    #     print(f"Ejecutando en modo {'Hiperparámetros' if hiperparam_search else 'Comité'} "
    #           "(secuencial a nivel de instancia, paralelización interna si aplica).")

    #     for task in instance:
    #         i, path, config = task
    #         result = run_multiple_instances((i, global_timestamp, config, path))
    # else:
    #     # Ejecución paralela
    #     print("Ejecutando en modo Estándar (paralelizado a nivel de instancia).")
    #     num_processes = multiprocessing.cpu_count()
    #     print(f"Iniciando la ejecución paralela en {num_processes} procesos...")
    #     print(f"configuración global del algoritmo: {config['algorithm']}")

    #     # Adaptar tasks para incluir timestamp
    #     tasks = [(i, global_timestamp, config, path) for (i, path, config) in instance]

    #     with multiprocessing.Pool(processes=num_processes) as pool:
    #         results = pool.map(run_single_instance, tasks)

    # print("Todas las operaciones completadas. Revisa las carpetas 'output' para los logs y resultados.")

# def create_main_execution_function():
#     """
#     Función principal mejorada que procesa múltiples instancias
#     """
    
#     def main_improved(algorithm='grasp', mode='committee', search_type='grid', 
#                      num_runs=5, num_processes=None, output_dir='output'):
#         """
#         Función principal mejorada para ejecutar experimentos
        
#         Args:
#             algorithm: 'grasp', 'genetic', 'tabu'
#             mode: 'committee' o 'hyperparameters'
#             search_type: 'grid', 'random', 'bayesian' (solo para hyperparameters)
#             num_runs: Número de ejecuciones por configuración
#             num_processes: Número de procesos paralelos
#             output_dir: Directorio de salida base
#         """
        
#         # configurar logging
#         logging.basicconfig(
#             level=logging.INFO,
#             format='%(asctime)s - %(levelname)s - %(message)s'
#         )
#         logger = logging.getLogger('main')
        
#         # Cargar instancias
#         instances = load_testing_instances()
        
#         if not instances:
#             logger.error("No se encontraron instancias para procesar")
#             return
        
#         logger.info(f"Cargadas {len(instances)} instancias")
        
#         # Crear directorio de salida con timestamp
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         result_base_dir = os.path.join(output_dir, f"{algorithm}_{mode}_{timestamp}")
#         os.makedirs(result_base_dir, exist_ok=True)
        
#         if mode == 'committee':
#             # Ejecutar comités con parámetros por defecto
#             if algorithm == 'grasp':
#                 # Parámetros por defecto para GRASP
#                 algorithm_params = []
#                 for i, instance in enumerate(instances):
#                     params = {
#                         'algorithm_type': 'grasp',
#                         'params': instance,
#                         'alpha': 0.3,
#                         'frac_neighbors': 4,
#                         'problem': 'P2',
#                         'max_iter': None
#                     }
#                     algorithm_params.append(params)
                
#             elif algorithm == 'genetic':
#                 # Parámetros por defecto para Genetic
#                 algorithm_params = []
#                 for i, instance in enumerate(instances):
#                     params = {
#                         'algorithm_type': 'genetic',
#                         'params': instance,
#                         'generations': 50,
#                         'mutation_rate': 0.05,
#                         'crossover_rate': 0.95,
#                         'tournament': 7,
#                         'inicializacion': 'random',
#                         'problem': 'P2'
#                     }
#                     algorithm_params.append(params)
                
#             elif algorithm == 'tabu':
#                 # Parámetros por defecto para Tabu
#                 algorithm_params = []
#                 for i, instance in enumerate(instances):
#                     params = {
#                         'algorithm_type': 'tabu',
#                         'params': instance,
#                         'tabu_tenure': 0.25,
#                         'inicializacion': 'random',
#                         'max_iter': 15,
#                         'gamma_f': 0.5,
#                         'gamma_q': 0.5,
#                         'time_limit': 100,
#                         'problem': 'P2'
#                     }
#                     algorithm_params.append(params)
            
#             # Ejecutar comités
#             results = run_committee_multiple_instances(
#                 algorithm_params, instances, result_base_dir,
#                 algorithm, num_runs, num_processes
#             )
            
#         elif mode == 'hyperparameters':
#             # Búsqueda de hiperparámetros
#             if algorithm == 'grasp':
#                 param_combinations = generate_grasp_param_combinations(search_type)
#             elif algorithm == 'genetic':
#                 param_combinations = generate_genetic_param_combinations(search_type)
#             elif algorithm == 'tabu':
#                 param_combinations = generate_tabu_param_combinations(search_type)
#             else:
#                 raise ValueError(f"Algoritmo desconocido: {algorithm}")
            
#             # Ejecutar búsqueda de hiperparámetros
#             results = run_hyperparameter_search_multiple_instances(
#                 algorithm, param_combinations, instances,
#                 result_base_dir, num_runs, num_processes
#             )
        
#         else:
#             raise ValueError(f"Modo desconocido: {mode}")
        
#         # Generar reporte final
#         generate_final_report(results, result_base_dir, algorithm, mode)
        
#         logger.info(f"Experimento completado. Resultados en: {result_base_dir}")
#         return results
    
#     return main_improved