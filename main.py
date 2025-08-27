import logging
import multiprocessing
import os
from datetime import datetime
import time

from Comittee_agent import run_genetic_committee, run_genetic_once, run_grasp_committee, run_grasp_hyperparam_search, run_grasp_once, run_tabu_committee, run_tabu_hyperparam_search, run_tabu_once
from algorithms.MILP import milp_ceflp as MILP

from structure.create_instances import map_R
from structure.instances import readInstance
from structure.pickup import get_dict_distances, get_dict_pickuppoints, get_pickuppoints, get_max_from_nested_dict

# C:\Users\Andrea\Documents\TFM\.venv\Scripts\Activate.ps1
# source venv_ceflp/bin/activate

CONFIG = {
    'save_results': True,
    'problem': 'P2', # Puede ser 'P1' o 'P2'
    'algorithm': 'TABU',  # Puede ser 'GRASP' o 'MILP' o 'TABU' o 'GENETIC'
    'optimization_solver': 'CPLEX',  # Puede ser 'CBC', 'GLPK' o 'CPLEX'
    'inicialization': 'random',  # Puede ser 'random', 'kmeans' o 'greedy'
    'comitee': False,  # Si se usa comité GRASP
    'hyperparam_search': False,  # Si se busca la mejor combinación de hiperparámetros
    'type_search': 'random',  # Tipo de búsqueda de hiperparámetros, puede ser 'grid', 'random' o 'bayesian'
    'alpha': 0.65,  # Parámetro de aleatoriedad para GREEDY
    'frac_neighbors': 3,  # Fracción a dividir el total de clientes para obtener el número de vecinos a generar por iteración
    'tabu_tenure': 0.35,  # Tenencia para el algoritmo Tabu
    'time_limit': 300,  # Límite de tiempo en segundos para la resolución del MILP
    'max_iter': None,  # Máximo número de iteraciones para el algoritmo Tabu
}

def setup_logger(log_file_path):
    """
    Configura un logger para escribir mensajes en un archivo y en la consola.
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

def get_coordinates(nodes):
    """
    Obtiene las coordenadas de los nodos en formato (x, y) a partir del diccionario de nodos.
    
    Parámetros:
    - nodes: Diccionario con la información de los nodos.
    
    Retorna:
    - Una lista de tuplas con las coordenadas (x, y) de cada nodo.
    """
    return [(int(value['x']), int(value['y'])) for value in nodes.values()]

def create_params(path, instance: int) -> dict:
    """
    Los conjuntos y parámetros son:
    I: lista de clientes
    J: lista de instalaciones
    K: lista de candidatos a puntos de recogida
    K_i: diccionario que para cada cliente i (en I) da el subconjunto Ki (lista de k en K)
    I_k: diccionario que para cada candidato k (en K) da el subconjunto I_k de clientes que pueden ir a k
    h: diccionario de demanda para cada i en I
    d_ij: diccionario doble para distancia/costo entre i en I y j en J
    d_kj: diccionario doble para distancia/costo entre k en K y j en J
    p: número de instalaciones a abrir
    t: número de puntos de recogida a abrir
    M_param: cota superior, por ejemplo, sum(h[i] for i in I)
    """

    # Leer la instancia
    instance_dict = readInstance(path)
    n = instance_dict['n']
    p = instance_dict['p']
    t = instance_dict['t']
    h = [int(value['demand']) for _, value in instance_dict['nodes'].items()]
    M_param = sum(h[i] for i in range(n))

    I = [i+1 for i in range(n)]
    J = I
    d_ij = instance_dict['d']
    dist_max = get_max_from_nested_dict(d_ij)
    R = round(dist_max*map_R(instance+1), 3)
    dist_pickuppoints = get_pickuppoints(R=R, nodes=instance_dict['nodes'])
    pickuppoints = [(round(float(x), 3), round(float(y), 3)) for x, y in dist_pickuppoints]

    # Para evitar problemas luego en la formulación 3-indices K no empezará en uno, sino en |I|
    K = [k+n+1 for k in range(len(pickuppoints))]
    # print(K)
    K_i = get_dict_pickuppoints(customers_dict=instance_dict['nodes'], pickups_list=pickuppoints, R=R, type="customer")
    I_k = get_dict_pickuppoints(customers_dict=instance_dict['nodes'], pickups_list=pickuppoints, R=R, type="candidate")
    # print(I_k)
    
    d_kj = get_dict_distances(I=instance_dict['nodes'], K=dist_pickuppoints)
    nodes_list = get_coordinates(instance_dict['nodes'])
    return {
        'I': I,
        'J': J,
        'K': K,
        'K_i': K_i,
        'I_k': I_k,
        'h': h,
        'd_ij': d_ij,
        'd_kj': d_kj,
        'p': p,
        't': t,
        'R': R,
        'M_param': M_param,
        'nodes': nodes_list,
        'pickuppoints': pickuppoints,
    }

def executeInstance(
                path, 
                instance: int, 
                logger: logging, 
                result_dir, 
                alpha: float = 0.3, 
                frac_neighbors: int = 4, 
                problem: str = 'P1', 
                algorithm: str = "GRASP", 
                optimizer: str = "CBC", 
                inicialization: str = "random", 
                tabu_tenure: float = 0.25,
                comitee: bool = False, 
                hiperparam_search: bool = False,
                type_search: str = 'grid',  
                num_runs: int = 4,
                time_limit = None,
                max_iter: int = 10
            ):
    start_time = time.time()
    params = create_params(path, instance)
    logger.info(f"Tiempo en crear parámetros {time.time()-start_time}")
    if algorithm == "GRASP":
        if hiperparam_search:
            run_grasp_hyperparam_search(params=params, instance=instance, result_dir=result_dir, logger=logger, problem=problem, num_runs=num_runs, type_search=type_search)
        elif comitee:
            run_grasp_committee(params = params, instance=instance, alpha=alpha, frac_neighbors=frac_neighbors, result_dir=result_dir, problem=problem, max_iter=max_iter, logger=logger, num_runs=num_runs)
        else:
            run_grasp_once(params = params, instance=instance, alpha=alpha, frac_neighbors=frac_neighbors, result_dir=result_dir, problem=problem, max_iter=max_iter, logger=logger)
        
    elif algorithm == "MILP":
        MILP(problem_type=problem, optimizer=optimizer, params = params, instance=instance, instance_problem="p1", result_dir=result_dir, time_limit=time_limit, logger=logger)

    elif algorithm == "TABU":
        if hiperparam_search:
            run_tabu_hyperparam_search(params=params, instance=instance, result_dir=result_dir, problem=problem, logger=logger, time_limit=time_limit, num_runs=num_runs, type_search=type_search)
        elif comitee:
            run_tabu_committee(params=params, instance=instance, tabu_tenure=tabu_tenure, result_dir=result_dir,  problem=problem,logger=logger, inicialization=inicialization, time_limit=time_limit, num_runs=num_runs)
        else:
            run_tabu_once(params=params, instance=instance, tabu_tenure=tabu_tenure, result_dir=result_dir,  problem=problem,logger=logger, inicialization=inicialization, time_limit=time_limit)

    elif algorithm == "GENETIC":
        if comitee:
            run_genetic_committee(params=params, instance=instance, result_dir=result_dir,  problem=problem, inicializacion=inicialization, logger=logger)
        else:
            run_genetic_once(params=params, instance=instance, result_dir=result_dir,  problem=problem, inicializacion=inicialization, logger=logger)

# --- Función adaptada para la paralelización ---
def run_single_instance(args_tuple):

    i, global_timestamp, current_config = args_tuple

    # Extraer los parámetros de configuración comunes del 'current_config'
    algorithm = current_config['algorithm']
    optimizer = current_config['optimization_solver']
    problem = current_config['problem']
    type_search = current_config.get('type_search', 'grid')
    tabu_tenure = current_config.get('tabu_tenure', 0.25)
    inicialization = current_config['inicialization']
    alpha = current_config.get('alpha', 0.3)
    frac_neighbors = current_config.get('frac_neighbors', 4)
    time_limit = current_config['time_limit']
    num_runs = current_config.get('num_runs', 4)
    max_iter = current_config['max_iter']
    hiperparam_search = current_config.get('hyperparam_search', False)
    comitee = current_config.get('comitee', False)

    instance_name = f"i{i+1}"
    path = f"instances/p1/{instance_name}.txt"

    # Determinar el directorio de resultados
    if algorithm == "MILP":
        result_dir = f"output/{problem}/{algorithm}/{optimizer}/{global_timestamp}"
    elif algorithm == "GRASP":
        if hiperparam_search:
            result_dir = f"output/{problem}/{algorithm}/hyperparam_search/{type_search}/{global_timestamp}"
        elif comitee:
            result_dir = f"output/{problem}/{algorithm}/comitee/{global_timestamp}"
        else:
            result_dir = f"output/{problem}/{algorithm}/once/{global_timestamp}"
    elif algorithm == "TABU":
        if hiperparam_search:
            result_dir = f"output/{problem}/{algorithm}/hyperparam_search/{type_search}/{global_timestamp}"
        elif comitee:
            result_dir = f"output/{problem}/{algorithm}/comitee/{inicialization}/{global_timestamp}"
        else:
            result_dir = f"output/{problem}/{algorithm}/once/{inicialization}/{global_timestamp}"
    elif algorithm == "GENETIC":
            if comitee:
                result_dir = f"output/{problem}/{algorithm}/comitee/{inicialization}/{global_timestamp}"
            else:
                result_dir = f"output/{problem}/{algorithm}/{inicialization}/{global_timestamp}"

    os.makedirs(result_dir, exist_ok=True)
    log_filename = os.path.join(result_dir, f"execution_{instance_name}_{global_timestamp}.log")
    instance_logger = setup_logger(log_filename) # Cada instancia, su propio logger

    instance_logger.info(f"--- Inicio de la ejecución para {instance_name}: {global_timestamp} ---")
    print(f"  [Proceso {os.getpid()}] Usando el solver: {optimizer}" if algorithm == "MILP" else f"  [Proceso {os.getpid()}] Usando {algorithm}")

    try:
        executeInstance(
            path=path,
            problem=problem,
            algorithm=algorithm,
            optimizer=optimizer,
            inicialization=inicialization,
            alpha=alpha,
            frac_neighbors=frac_neighbors,
            tabu_tenure=tabu_tenure,
            instance=i,
            comitee=comitee,
            hiperparam_search=hiperparam_search,
            type_search=type_search,
            result_dir=result_dir,
            time_limit=time_limit,
            num_runs=num_runs,
            max_iter=max_iter,
            logger=instance_logger,
        )
        instance_logger.info(f"--- Fin de la ejecución para {instance_name} ---")
        return f"Instancia {instance_name} completada con éxito."
    except Exception as e:
        instance_logger.error(f"Error procesando instancia {instance_name}: {e}")
        return f"Error en instancia {instance_name}: {e}"


# --- Bloque principal para la ejecución paralela ---
if __name__ == "__main__":
    global_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    comitee = CONFIG.get('comitee', False)
    hiperparam_search = CONFIG.get('hyperparam_search', False)

    # El algoritmo se ejecuta de forma secuencial si es una búsqueda de hiperparámetros o comité,
    # ya que estas funciones ya gestionan su propia paralelización interna o series de ejecuciones.
    if hiperparam_search or comitee: 
        print(f"Ejecutando en modo {'Hiperparámetros' if hiperparam_search else 'Comité'} (secuencial a nivel de instancia, paralelización interna si aplica).")

        for i in range(0, 10):
            result = run_single_instance((i, global_timestamp, CONFIG))
    else:
        # Modo de ejecución de instancia a instancia, paralelizado
        print(f"Ejecutando en modo Estándar (paralelizado a nivel de instancia).")
        tasks = []
        for i in range(0, 15):
            instance_name = f"i{i+1}"
            path = f"instances/p1/{instance_name}.txt"
            tasks.append((i, global_timestamp, CONFIG))

        num_processes = multiprocessing.cpu_count()
        print(f"Iniciando la ejecución paralela en {num_processes} procesos...")
        print(f"Configuración global del algoritmo: {CONFIG['algorithm']}")

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(run_single_instance, tasks)

    print(f"Todas las operaciones completadas. Revisa las carpetas 'output' para los logs y resultados.")
