from math import sqrt
import time
import os
import traceback
import pandas as pd
import numpy as np
from itertools import product
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

from algorithms.GRASP import GRASPSearch
from algorithms.GeneticAlgorithm import GeneticSearch
from algorithms.TABU import TabuSearch

import random
random.seed(42)

# Imports para Bayesian Optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei
    BAYESIAN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-optimize no está instalado. Instalar con: pip install scikit-optimize")
    BAYESIAN_AVAILABLE = False

def setup_worker_logger():
    """Configura un logger para cada proceso worker"""
    logger = logging.getLogger(f'worker_{os.getpid()}')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - Worker %(process)d - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def run_single_grasp_execution(args):
    """Ejecuta una sola ejecución de GRASP - función para paralelización plana"""
    params, instance, base_result_dir, alpha, frac_neighbors, problem, max_iter, run_idx, combination_name = args
    
    worker_logger = setup_worker_logger()
    
    # Crear directorio específico para esta ejecución
    if combination_name:
        result_dir = f"{base_result_dir}/{combination_name}/Comite_{run_idx+1}"
    else:
        result_dir = f"{base_result_dir}/Comite_{run_idx+1}"
    
    os.makedirs(result_dir, exist_ok=True)
    
    worker_logger.info(f"[GRASP WORKER] Iniciando ejecución {run_idx + 1} - {combination_name if combination_name else 'Comité simple'}")
    
    grasp = GRASPSearch(
        params=params, 
        instance=instance, 
        result_dir=result_dir, 
        alpha=alpha, 
        frac_neighbors=frac_neighbors, 
        logger=worker_logger, 
        problem=problem, 
        max_iter=max_iter
    )
    grasp.run()
    
    worker_logger.info(f"[GRASP WORKER] Ejecución {run_idx + 1} completada - Costo: {grasp.best_solution.cost:.2f}")
    
    return {
        'solution': deepcopy(grasp.best_solution),
        'run_idx': run_idx,
        'combination_name': combination_name,
        'alpha': alpha,
        'frac_neighbors': frac_neighbors
    }

def run_single_genetic_execution(args):
    """Ejecuta una sola ejecución de GENETIC - función para paralelización plana"""
    params, instance, base_result_dir, problem, inicialization, generations, mutation_rate, crossover_rate, tournament, run_idx, combination_name = args
    
    worker_logger = setup_worker_logger()
    
    # Crear directorio específico para esta ejecución
    if combination_name:
        result_dir = f"{base_result_dir}/{combination_name}/Comite_{run_idx+1}"
    else:
        result_dir = f"{base_result_dir}/Comite_{run_idx+1}"
    
    os.makedirs(result_dir, exist_ok=True)
    
    worker_logger.info(f"[GENETIC WORKER] Iniciando ejecución {run_idx + 1} - {combination_name if combination_name else 'Comité simple'}")
    
    genetic = GeneticSearch(
        params=params, 
        instance=instance, 
        problem=problem, 
        result_dir=result_dir, 
        inicializacion=inicialization, 
        generations=generations, 
        mutation_rate=mutation_rate, 
        crossover_rate=crossover_rate, 
        tournament=tournament, 
        logger=worker_logger)
    genetic.run()
    
    worker_logger.info(f"[GENETIC WORKER] Ejecución {run_idx + 1} completada - Costo: {genetic.best_individual.cost:.2f}")
    
    return {
        'solution': deepcopy(genetic.best_individual),
        'run_idx': run_idx,
        'combination_name': combination_name,
        'generations': generations,
        'mutation_rate': mutation_rate,
        'crossover_rate': crossover_rate,
        'tournament': tournament,
        'inicializacion': inicialization
    }

def run_single_tabu_execution(args):
    """Ejecuta una sola ejecución de TABU - función para paralelización plana"""
    params, instance, base_result_dir, problem, inicialization, tabu_tenure, max_iter, run_idx, combination_name = args
    
    worker_logger = setup_worker_logger()
    
    # Crear directorio específico para esta ejecución
    if combination_name:
        result_dir = f"{base_result_dir}/{combination_name}/Comite_{run_idx+1}"
    else:
        result_dir = f"{base_result_dir}/Comite_{run_idx+1}"
    
    os.makedirs(result_dir, exist_ok=True)
    
    worker_logger.info(f"[TABU WORKER] Iniciando ejecución {run_idx + 1} - {combination_name if combination_name else 'Comité simple'}")
    
    tabu = TabuSearch(
        params=params, 
        inicialization=inicialization, 
        instance=instance, 
        problem=problem, 
        result_dir=result_dir, 
        tabu_tenure=tabu_tenure, 
        max_iter=max_iter, 
        logger=worker_logger
    )
    tabu.run()
    
    worker_logger.info(f"[TABU WORKER] Ejecución {run_idx + 1} completada - Costo: {tabu.best_solution.cost:.2f}")
    
    return {
        'solution': deepcopy(tabu.best_solution),
        'run_idx': run_idx,
        'combination_name': combination_name,
        'tabu_tenure': tabu_tenure,
        'inicializacion': inicialization
    }

def run_grasp_committee_parallel(params, instance, result_dir, logger, alpha=0.3, frac_neighbors=4, problem="P2", max_iter=None, num_runs=5, num_processes=None):
    """Versión paralelizada del comité GRASP usando paralelización plana"""
    if num_processes is None:
        num_processes = min(cpu_count(), num_runs)
    
    logger.info(f"\n--- Ejecutando comité GRASP PARALELO con {num_runs} miembros usando {num_processes} procesos ---\n")
    
    start = time.time()
    
    # Preparar argumentos para cada ejecución
    args_list = [
        (params, instance, result_dir, alpha, frac_neighbors, problem, max_iter, idx, None)
        for idx in range(num_runs)
    ]
    
    # Ejecutar todas las ejecuciones en paralelo
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(run_single_grasp_execution, args_list))
    
    # Procesar resultados
    solutions = []
    costs = []
    
    for result in results:
        solutions.append(result['solution'])
        costs.append(result['solution'].cost)
        logger.info(f"[COMITÉ] Ejecución {result['run_idx'] + 1}: Costo {result['solution'].cost:.2f}")
    
    best_index = int(np.argmin(costs))
    best_solution = solutions[best_index]
    end = time.time()
    total_time = end - start
    
    logger.info(f"\n[COMITÉ PARALELO] Mejor costo encontrado: {best_solution.cost:.2f}")
    logger.info(f"[COMITÉ PARALELO] Tiempo de mejor solución: {best_solution.time:.2f} segundos")
    logger.info(f"[COMITÉ PARALELO] Media de costos: {np.mean(costs):.2f}")
    logger.info(f"[COMITÉ PARALELO] Desviación estándar: {np.std(costs):.2f}")
    logger.info(f"[COMITÉ PARALELO] Tiempo total: {total_time:.2f} segundos\n")
    
    num_neighbors = max(1, len(params['I']) // frac_neighbors)
    save_solution(sol=best_solution, time=total_time, costs=costs, alpha=alpha, num_neighbors=num_neighbors, instance=instance, result_dir=result_dir)
    
    return best_solution

def run_tabu_committee_parallel(params, instance, result_dir, logger, inicialization="random", max_iter=50, num_runs=5, tabu_tenure=7, problem="P2", num_processes=None):
    """Versión paralelizada del comité TABU usando paralelización plana"""
    if num_processes is None:
        num_processes = min(cpu_count(), num_runs)
    
    logger.info(f"\n--- Ejecutando comité TABÚ PARALELO con {num_runs} miembros usando {num_processes} procesos ---\n")
    
    start = time.time()
    
    # Preparar argumentos para cada ejecución
    args_list = [
        (params, instance, result_dir, problem, inicialization, tabu_tenure, max_iter, idx, None)
        for idx in range(num_runs)
    ]
    
    # Ejecutar todas las ejecuciones en paralelo
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(run_single_tabu_execution, args_list))
    
    # Procesar resultados
    solutions = []
    costs = []
    
    for result in results:
        solutions.append(result['solution'])
        costs.append(result['solution'].cost)
        logger.info(f"[COMITÉ] Ejecución {result['run_idx'] + 1}: Costo {result['solution'].cost:.2f}")
    
    best_index = int(np.argmin(costs))
    best_solution = solutions[best_index]
    end = time.time()
    total_time = end - start
    
    logger.info(f"\n[COMITÉ PARALELO] Mejor costo encontrado: {best_solution.cost:.2f}")
    logger.info(f"[COMITÉ PARALELO] Tiempo de mejor solución: {best_solution.time:.2f} segundos")
    logger.info(f"[COMITÉ PARALELO] Media de costos: {np.mean(costs):.2f}")
    logger.info(f"[COMITÉ PARALELO] Desviación estándar: {np.std(costs):.2f}")
    logger.info(f"[COMITÉ PARALELO] Tiempo total: {total_time:.2f} segundos\n")
    
    save_solution(sol=best_solution, time=total_time, costs=costs, instance=instance, result_dir=result_dir)
    
    return best_solution

def run_genetic_committee_parallel(params, instance, result_dir, logger, inicializacion="random", generations=50, num_runs=5, mutation_rate=0.05, crossover_rate=0.95, tournament=7, problem="P2", num_processes=None):
    """Versión paralelizada del comité GENETIC usando paralelización plana"""
    if num_processes is None:
        num_processes = min(cpu_count(), num_runs)
    
    logger.info(f"\n--- Ejecutando comité GENETIC PARALELO con {num_runs} miembros usando {num_processes} procesos ---\n")
    
    start = time.time()
    
    # Preparar argumentos para cada ejecución
    args_list = [
        (params, instance, result_dir, problem, inicializacion, generations, mutation_rate, crossover_rate, tournament, idx, None)
        for idx in range(num_runs)
    ]
    
    # Ejecutar todas las ejecuciones en paralelo
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(run_single_genetic_execution, args_list))
    
    # Procesar resultados
    solutions = []
    costs = []
    
    for result in results:
        solutions.append(result['solution'])
        costs.append(result['solution'].cost)
        logger.info(f"[COMITÉ] Ejecución {result['run_idx'] + 1}: Costo {result['solution'].cost:.2f}")
    
    best_index = int(np.argmin(costs))
    best_solution = solutions[best_index]
    end = time.time()
    total_time = end - start
    
    logger.info(f"\n[COMITÉ PARALELO] Mejor costo encontrado: {best_solution.cost:.2f}")
    logger.info(f"[COMITÉ PARALELO] Tiempo de mejor solución: {best_solution.time:.2f} segundos")
    logger.info(f"[COMITÉ PARALELO] Media de costos: {np.mean(costs):.2f}")
    logger.info(f"[COMITÉ PARALELO] Desviación estándar: {np.std(costs):.2f}")
    logger.info(f"[COMITÉ PARALELO] Tiempo total: {total_time:.2f} segundos\n")
    
    save_solution(sol=best_solution, time=total_time, costs=costs, instance=instance, result_dir=result_dir)
    
    return best_solution

def run_grasp_hyperparam_grid_search(params, instance, result_dir, logger, problem="P2", num_runs=8, max_iter=None, num_processes=None):
    """Búsqueda de hiperparámetros GRASP con paralelización plana completa"""
    alphas = [0.1, 0.3, 0.5, 0.7]
    frac_neighbors_list = [2, 4, 6]
    combinations = list(product(alphas, frac_neighbors_list))
    
    # Calcular el total de ejecuciones individuales
    total_executions = len(combinations) * num_runs
    
    if num_processes is None:
        num_processes = min(cpu_count(), total_executions)
    
    logger.info(f"\n=== Búsqueda PARALELA de hiperparámetros GRASP ===")
    logger.info(f"Combinaciones: {len(combinations)}, Ejecuciones por combinación: {num_runs}")
    logger.info(f"Total de ejecuciones: {total_executions}, Procesos: {num_processes}")
    
    start_time = time.time()
    
    # Preparar TODAS las ejecuciones individuales
    all_args = []
    for alpha, frac_neighbors in combinations:
        combination_name = f"alpha_{alpha}_frac_{frac_neighbors}"
        for run_idx in range(num_runs):
            all_args.append((
                params, instance, result_dir, alpha, frac_neighbors, 
                problem, max_iter, run_idx, combination_name
            ))
    
    # Ejecutar TODAS las ejecuciones en paralelo
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        all_results = list(executor.map(run_single_grasp_execution, all_args))
    
    # Procesar resultados por combinación
    combination_results = []
    results_by_combination = {}
    
    # Agrupar resultados por combinación
    for result in all_results:
        combo_key = (result['alpha'], result['frac_neighbors'])
        if combo_key not in results_by_combination:
            results_by_combination[combo_key] = []
        results_by_combination[combo_key].append(result)
    
    # Procesar cada combinación
    for (alpha, frac_neighbors), combo_results in results_by_combination.items():
        costs = [r['solution'].cost for r in combo_results]
        best_idx = np.argmin(costs)
        best_solution = combo_results[best_idx]['solution']
        
        logger.info(f"Combinación Alpha={alpha}, Frac_neighbors={frac_neighbors}: Mejor costo={best_solution.cost:.2f}")
        
        combination_results.append({
            "instance": instance+1,
            "alpha": alpha,
            "frac_neighbors": frac_neighbors,
            "best_cost": best_solution.cost,
            "solve_time": best_solution.time
        })
        
        # Guardar la mejor solución de esta combinación
        combo_dir = os.path.join(result_dir, f"alpha_{alpha}_frac_{frac_neighbors}")
        os.makedirs(combo_dir, exist_ok=True)
        num_neighbors = max(1, len(params['I']) // frac_neighbors)
        save_solution(
            sol=best_solution, 
            time=sum(r['solution'].time for r in combo_results), 
            costs=costs, 
            alpha=alpha, 
            num_neighbors=num_neighbors, 
            instance=instance, 
            result_dir=combo_dir
        )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Guardar resultados finales
    df = pd.DataFrame(combination_results)
    df = df.sort_values(by="best_cost")
    csv_path = os.path.join(result_dir, f"grasp_hyperparam_results_instance{instance+1}.csv")
    df.to_csv(csv_path, index=False)
    
    logger.info(f"\n=== Búsqueda PARALELA de hiperparámetros GRASP finalizada ===")
    logger.info(f"Tiempo total: {total_time:.2f} segundos")
    logger.info(f"Speedup estimado: {total_executions/num_processes:.1f}x")
    logger.info(f"Resultados guardados en: {csv_path}")
    
    return df

def run_tabu_hyperparam_grid_search(params, instance, result_dir, logger, problem="P2", num_runs=8, max_iter=None, num_processes=None):
    """Búsqueda de hiperparámetros TABU con paralelización plana completa"""
    num_clientes = len(params['I'])
    tabu_tenure_list = [int(0.15 * num_clientes), int(0.25 * num_clientes), int(0.5 * num_clientes), int(sqrt(num_clientes))]
    inicializacion_list = ["random", "greedy", "kmeans"]
    combinations = list(product(tabu_tenure_list, inicializacion_list))
    
    # Calcular el total de ejecuciones individuales
    total_executions = len(combinations) * num_runs
    
    if num_processes is None:
        num_processes = min(cpu_count(), total_executions)
    
    logger.info(f"\n=== Búsqueda PARALELA de hiperparámetros TABU ===")
    logger.info(f"Combinaciones: {len(combinations)}, Ejecuciones por combinación: {num_runs}")
    logger.info(f"Total de ejecuciones: {total_executions}, Procesos: {num_processes}")
    
    start_time = time.time()
    
    # Preparar TODAS las ejecuciones individuales
    all_args = []
    for tabu_tenure, inicializacion in combinations:
        combination_name = f"tabu_{tabu_tenure}_init_{inicializacion}"
        for run_idx in range(num_runs):
            all_args.append((
                params, instance, result_dir, problem, inicializacion, 
                tabu_tenure, max_iter, run_idx, combination_name
            ))
    
    # Ejecutar TODAS las ejecuciones en paralelo
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        all_results = list(executor.map(run_single_tabu_execution, all_args))
    
    # Procesar resultados por combinación
    combination_results = []
    results_by_combination = {}
    
    # Agrupar resultados por combinación
    for result in all_results:
        combo_key = (result['tabu_tenure'], result['inicializacion'])
        if combo_key not in results_by_combination:
            results_by_combination[combo_key] = []
        results_by_combination[combo_key].append(result)
    
    # Procesar cada combinación
    for (tabu_tenure, inicializacion), combo_results in results_by_combination.items():
        costs = [r['solution'].cost for r in combo_results]
        best_idx = np.argmin(costs)
        best_solution = combo_results[best_idx]['solution']
        
        logger.info(f"Combinación Tabu_tenure={tabu_tenure}, Inicialización={inicializacion}: Mejor costo={best_solution.cost:.2f}")
        
        combination_results.append({
            "instance": instance+1,
            "tabu_tenure": tabu_tenure,
            "inicializacion": inicializacion,
            "best_cost": best_solution.cost,
            "solve_time": best_solution.time
        })
        
        # Guardar la mejor solución de esta combinación
        combo_dir = os.path.join(result_dir, f"tabu_{tabu_tenure}_init_{inicializacion}")
        os.makedirs(combo_dir, exist_ok=True)
        save_solution(
            sol=best_solution, 
            time=sum(r['solution'].time for r in combo_results), 
            costs=costs, 
            instance=instance, 
            result_dir=combo_dir
        )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Guardar resultados finales
    df = pd.DataFrame(combination_results)
    df = df.sort_values(by="best_cost")
    csv_path = os.path.join(result_dir, f"tabu_hyperparam_results_instance{instance+1}.csv")
    df.to_csv(csv_path, index=False)
    
    logger.info(f"\n=== Búsqueda PARALELA de hiperparámetros TABU finalizada ===")
    logger.info(f"Tiempo total: {total_time:.2f} segundos")
    logger.info(f"Speedup estimado: {total_executions/num_processes:.1f}x")
    logger.info(f"Resultados guardados en: {csv_path}")
    
    return df

def run_grasp_hyperparam_random_search(params, instance, result_dir, logger, problem="P2", num_runs=8, max_iter=None, num_processes=None, num_random_combinations=10):
    """Búsqueda de hiperparámetros GRASP con paralelización plana completa"""
    combinations = set()
    while len(combinations) < num_random_combinations:
        alpha = round(random.uniform(0.0, 1.0), 2)  # con 2 decimales
        frac_neighbors = random.randint(2, 8)
        combinations.add((alpha, frac_neighbors))
    combinations = list(combinations)
    
    # Calcular el total de ejecuciones individuales
    total_executions = len(combinations) * num_runs
    
    if num_processes is None:
        num_processes = min(cpu_count(), total_executions)
    
    logger.info(f"\n=== Búsqueda PARALELA de hiperparámetros GRASP ===")
    logger.info(f"Combinaciones: {len(combinations)}, Ejecuciones por combinación: {num_runs}")
    logger.info(f"Total de ejecuciones: {total_executions}, Procesos: {num_processes}")
    
    start_time = time.time()
    
    # Preparar TODAS las ejecuciones individuales
    all_args = []
    for alpha, frac_neighbors in combinations:
        combination_name = f"alpha_{alpha}_frac_{frac_neighbors}"
        for run_idx in range(num_runs):
            all_args.append((
                params, instance, result_dir, alpha, frac_neighbors, 
                problem, max_iter, run_idx, combination_name
            ))
    
    # Ejecutar TODAS las ejecuciones en paralelo
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        all_results = list(executor.map(run_single_grasp_execution, all_args))
    
    # Procesar resultados por combinación
    combination_results = []
    results_by_combination = {}
    
    # Agrupar resultados por combinación
    for result in all_results:
        combo_key = (result['alpha'], result['frac_neighbors'])
        if combo_key not in results_by_combination:
            results_by_combination[combo_key] = []
        results_by_combination[combo_key].append(result)
    
    # Procesar cada combinación
    for (alpha, frac_neighbors), combo_results in results_by_combination.items():
        costs = [r['solution'].cost for r in combo_results]
        best_idx = np.argmin(costs)
        best_solution = combo_results[best_idx]['solution']
        
        logger.info(f"Combinación Alpha={alpha}, Frac_neighbors={frac_neighbors}: Mejor costo={best_solution.cost:.2f}")
        
        combination_results.append({
            "instance": instance+1,
            "alpha": alpha,
            "frac_neighbors": frac_neighbors,
            "best_cost": best_solution.cost,
            "solve_time": best_solution.time
        })
        
        # Guardar la mejor solución de esta combinación
        combo_dir = os.path.join(result_dir, f"alpha_{alpha}_frac_{frac_neighbors}")
        os.makedirs(combo_dir, exist_ok=True)
        num_neighbors = max(1, len(params['I']) // frac_neighbors)
        save_solution(
            sol=best_solution, 
            time=sum(r['solution'].time for r in combo_results), 
            costs=costs, 
            alpha=alpha, 
            num_neighbors=num_neighbors, 
            instance=instance, 
            result_dir=combo_dir
        )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Guardar resultados finales
    df = pd.DataFrame(combination_results)
    df = df.sort_values(by="best_cost")
    csv_path = os.path.join(result_dir, f"grasp_hyperparam_results_instance{instance+1}.csv")
    df.to_csv(csv_path, index=False)
    
    logger.info(f"\n=== Búsqueda PARALELA de hiperparámetros GRASP finalizada ===")
    logger.info(f"Tiempo total: {total_time:.2f} segundos")
    logger.info(f"Speedup estimado: {total_executions/num_processes:.1f}x")
    logger.info(f"Resultados guardados en: {csv_path}")
    
    return df

def run_tabu_hyperparam_random_search(params, instance, result_dir, logger, problem="P2", num_runs=8, max_iter=None, num_processes=None, num_random_combinations=10):
    """Búsqueda de hiperparámetros TABU con paralelización plana completa"""
    combinations = set()
    num_clientes = len(params['I'])
    init_strategies = ["random", "greedy", "kmeans"]

    while len(combinations) < num_random_combinations:
        # Tabu tenure aleatorio, entre 5 y 0.2 * n_clientes
        tabu_tenure = random.randint(int(0.25 * num_clientes), int(0.5 * num_clientes))
        
        # Asegurar que no sea mayor que num_clientes
        tabu_tenure = min(tabu_tenure, num_clientes)

        init_method = random.choice(init_strategies)

        combinations.add((tabu_tenure, init_method))

    combinations = list(combinations)
    
    # Calcular el total de ejecuciones individuales
    total_executions = len(combinations) * num_runs
    
    if num_processes is None:
        num_processes = min(cpu_count(), total_executions)
    
    logger.info(f"\n=== Búsqueda PARALELA de hiperparámetros TABU ===")
    logger.info(f"Combinaciones: {len(combinations)}, Ejecuciones por combinación: {num_runs}")
    logger.info(f"Total de ejecuciones: {total_executions}, Procesos: {num_processes}")
    
    start_time = time.time()
    
    # Preparar TODAS las ejecuciones individuales
    all_args = []
    for tabu_tenure, inicializacion in combinations:
        combination_name = f"tabu_{tabu_tenure}_init_{inicializacion}"
        for run_idx in range(num_runs):
            all_args.append((
                params, instance, result_dir, problem, inicializacion, 
                tabu_tenure, max_iter, run_idx, combination_name
            ))
    
    # Ejecutar TODAS las ejecuciones en paralelo
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        all_results = list(executor.map(run_single_tabu_execution, all_args))
    
    # Procesar resultados por combinación
    combination_results = []
    results_by_combination = {}
    
    # Agrupar resultados por combinación
    for result in all_results:
        combo_key = (result['tabu_tenure'], result['inicializacion'])
        if combo_key not in results_by_combination:
            results_by_combination[combo_key] = []
        results_by_combination[combo_key].append(result)
    
    # Procesar cada combinación
    for (tabu_tenure, inicializacion), combo_results in results_by_combination.items():
        costs = [r['solution'].cost for r in combo_results]
        best_idx = np.argmin(costs)
        best_solution = combo_results[best_idx]['solution']
        
        logger.info(f"Combinación Tabu_tenure={tabu_tenure}, Inicialización={inicializacion}: Mejor costo={best_solution.cost:.2f}")
        
        combination_results.append({
            "instance": instance+1,
            "tabu_tenure": tabu_tenure,
            "inicializacion": inicializacion,
            "best_cost": best_solution.cost,
            "solve_time": best_solution.time
        })
        
        # Guardar la mejor solución de esta combinación
        combo_dir = os.path.join(result_dir, f"tabu_{tabu_tenure}_init_{inicializacion}")
        os.makedirs(combo_dir, exist_ok=True)
        save_solution(
            sol=best_solution, 
            time=sum(r['solution'].time for r in combo_results), 
            costs=costs, 
            instance=instance, 
            result_dir=combo_dir
        )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Guardar resultados finales
    df = pd.DataFrame(combination_results)
    df = df.sort_values(by="best_cost")
    csv_path = os.path.join(result_dir, f"tabu_hyperparam_results_instance{instance+1}.csv")
    df.to_csv(csv_path, index=False)
    
    logger.info(f"\n=== Búsqueda PARALELA de hiperparámetros TABU finalizada ===")
    logger.info(f"Tiempo total: {total_time:.2f} segundos")
    logger.info(f"Speedup estimado: {total_executions/num_processes:.1f}x")
    logger.info(f"Resultados guardados en: {csv_path}")
    
    return df

def evaluate_grasp_hyperparams(alpha, frac_neighbors, params, instance, result_dir, logger, problem, num_runs, max_iter, num_processes):
    """Función objetivo para evaluación bayesiana de hiperparámetros GRASP"""
    # Convertir frac_neighbors de float a int (requerido por scikit-optimize)
    frac_neighbors = int(round(frac_neighbors))
    
    combination_name = f"alpha_{alpha:.3f}_frac_{frac_neighbors}"
    logger.info(f"[BAYESIAN] Evaluando: Alpha={alpha:.3f}, Frac_neighbors={frac_neighbors}")
    
    combo_dir = os.path.join(result_dir, "bayesian_search", combination_name)
    os.makedirs(combo_dir, exist_ok=True)
    
    # Preparar argumentos para todas las ejecuciones de esta combinación
    all_args = []
    for run_idx in range(num_runs):
        all_args.append((
            params, instance, combo_dir, alpha, frac_neighbors, 
            problem, max_iter, run_idx, combination_name
        ))
    
    # Ejecutar comité en paralelo
    with ProcessPoolExecutor(max_workers=min(num_processes, num_runs)) as executor:
        results = list(executor.map(run_single_grasp_execution, all_args))
    
    # Obtener el mejor resultado
    costs = [r['solution'].cost for r in results]
    best_idx = np.argmin(costs)
    best_cost = costs[best_idx]
    best_solution = results[best_idx]['solution']
    
    logger.info(f"[BAYESIAN] Alpha={alpha:.3f}, Frac_neighbors={frac_neighbors} -> Mejor costo: {best_cost:.2f}")
    
    # Guardar solución
    num_neighbors = max(1, len(params['I']) // frac_neighbors)
    save_solution(
        sol=best_solution, 
        time=sum(r['solution'].time for r in results), 
        costs=costs, 
        alpha=alpha, 
        num_neighbors=num_neighbors, 
        instance=instance, 
        result_dir=combo_dir
    )
    
    return best_cost

def run_grasp_hyperparam_bayesian_search(params, instance, result_dir, logger, problem="P2", num_runs=8, max_iter=None, 
                             num_processes=None, n_calls=20, n_initial_points=5, random_state=42):
    """Búsqueda bayesiana de hiperparámetros para GRASP"""
    
    if not BAYESIAN_AVAILABLE:
        logger.error("scikit-optimize no está disponible. Ejecutando búsqueda en grilla como fallback.")
        return run_grasp_hyperparam_random_search(params, instance, result_dir, logger, problem, num_runs, max_iter, num_processes)
    
    if num_processes is None:
        num_processes = min(cpu_count(), num_runs)
    
    logger.info(f"\n=== Búsqueda BAYESIANA de hiperparámetros GRASP ===")
    logger.info(f"Evaluaciones totales: {n_calls}, Puntos iniciales aleatorios: {n_initial_points}")
    logger.info(f"Ejecuciones por evaluación: {num_runs}, Procesos: {num_processes}")
    
    # Definir el espacio de búsqueda
    dimensions = [
        Real(0.05, 0.95, name='alpha'),           # Alpha entre 0.05 y 0.95
        Integer(2, 8, name='frac_neighbors')      # Frac_neighbors entre 2 y 8
    ]
    
    # Función objetivo con decorador para argumentos nombrados
    @use_named_args(dimensions)
    def objective(**params_dict):
        return evaluate_grasp_hyperparams(
            alpha=params_dict['alpha'],
            frac_neighbors=params_dict['frac_neighbors'],
            params=params,
            instance=instance,
            result_dir=result_dir,
            logger=logger,
            problem=problem,
            num_runs=num_runs,
            max_iter=max_iter,
            num_processes=num_processes
        )
    
    start_time = time.time()
    
    # Ejecutar optimización bayesiana
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        acq_func='EI',  # Expected Improvement
        random_state=random_state,
        n_jobs=1  # Paralelización se maneja internamente
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Procesar resultados
    all_results = []
    for i, (x_val, y_val) in enumerate(zip(result.x_iters, result.func_vals)):
        all_results.append({
            "instance": instance+1,
            "iteration": i+1,
            "alpha": x_val[0],
            "frac_neighbors": int(x_val[1]),
            "best_cost": y_val,
            "is_best": i == np.argmin(result.func_vals)
        })
    
    # Obtener mejor resultado
    best_alpha, best_frac_neighbors = result.x
    best_frac_neighbors = int(best_frac_neighbors)
    best_cost = result.fun
    
    logger.info(f"\n=== Resultado de Búsqueda Bayesiana GRASP ===")
    logger.info(f"Mejor combinación encontrada:")
    logger.info(f"  Alpha: {best_alpha:.3f}")
    logger.info(f"  Frac_neighbors: {best_frac_neighbors}")
    logger.info(f"  Mejor costo: {best_cost:.2f}")
    logger.info(f"Tiempo total: {total_time:.2f} segundos")
    logger.info(f"Mejora estimada vs grid search: {((4*3*num_runs) - n_calls*num_runs) / (4*3*num_runs):.1%} menos evaluaciones")
    
    # Guardar resultados
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(result_dir, f"grasp_bayesian_results_instance{instance+1}.csv")
    df.to_csv(csv_path, index=False)
    
    # Guardar también el historial de optimización
    history_path = os.path.join(result_dir, f"grasp_bayesian_history_instance{instance+1}.csv")
    history_df = pd.DataFrame({
        'iteration': range(1, len(result.func_vals)+1),
        'alpha': [x[0] for x in result.x_iters],
        'frac_neighbors': [int(x[1]) for x in result.x_iters],
        'cost': result.func_vals,
        'cumulative_best': np.minimum.accumulate(result.func_vals)
    })
    history_df.to_csv(history_path, index=False)
    
    logger.info(f"Resultados guardados en: {csv_path}")
    logger.info(f"Historial guardado en: {history_path}")
    
    return df

def evaluate_tabu_hyperparams(tabu_tenure, inicializacion, params, instance, result_dir, logger, problem, num_runs, max_iter, num_processes):
    """Función objetivo para evaluación bayesiana de hiperparámetros TABU"""
    # Convertir tabu_tenure de float a int (requerido por scikit-optimize)
    tabu_tenure = int(round(tabu_tenure))
    
    combination_name = f"tabu_{tabu_tenure}_init_{inicializacion}"
    logger.info(f"[BAYESIAN] Evaluando: Tabu_tenure={tabu_tenure}, Inicialización={inicializacion}")
    
    combo_dir = os.path.join(result_dir, "bayesian_search", combination_name)
    os.makedirs(combo_dir, exist_ok=True)
    
    # Preparar argumentos para todas las ejecuciones de esta combinación
    all_args = []
    for run_idx in range(num_runs):
        all_args.append((
            params, instance, combo_dir, problem, inicializacion, 
            tabu_tenure, max_iter, run_idx, combination_name
        ))
    
    # Ejecutar comité en paralelo
    with ProcessPoolExecutor(max_workers=min(num_processes, num_runs)) as executor:
        results = list(executor.map(run_single_tabu_execution, all_args))
    
    # Obtener el mejor resultado
    costs = [r['solution'].cost for r in results]
    best_idx = np.argmin(costs)
    best_cost = costs[best_idx]
    best_solution = results[best_idx]['solution']
    
    logger.info(f"[BAYESIAN] Tabu_tenure={tabu_tenure}, Inicialización={inicializacion} -> Mejor costo: {best_cost:.2f}")
    
    # Guardar solución
    save_solution(
        sol=best_solution, 
        time=sum(r['solution'].time for r in results), 
        costs=costs, 
        instance=instance, 
        result_dir=combo_dir
    )
    
    return best_cost

def run_tabu_hyperparam_bayesian_search(params, instance, result_dir, logger, problem="P2", num_runs=8, max_iter=None, 
                            num_processes=None, n_calls=25, n_initial_points=8, random_state=42):
    """Búsqueda bayesiana de hiperparámetros para TABU"""
    
    if not BAYESIAN_AVAILABLE:
        logger.error("scikit-optimize no está disponible. Ejecutando búsqueda en grilla como fallback.")
        return run_tabu_hyperparam_search(params, instance, result_dir, logger, problem, num_runs, max_iter, num_processes)
    
    if num_processes is None:
        num_processes = min(cpu_count(), num_runs)
    
    # Calcular rangos adaptativos basados en el tamaño del problema
    num_clientes = len(params['I'])
    min_tenure = 3
    max_tenure = max(20, int(0.3 * num_clientes))
    
    logger.info(f"\n=== Búsqueda BAYESIANA de hiperparámetros TABU ===")
    logger.info(f"Evaluaciones totales: {n_calls}, Puntos iniciales aleatorios: {n_initial_points}")
    logger.info(f"Ejecuciones por evaluación: {num_runs}, Procesos: {num_processes}")
    logger.info(f"Rango Tabu tenure: [{min_tenure}, {max_tenure}]")
    
    # Definir el espacio de búsqueda
    dimensions = [
        Integer(min_tenure, max_tenure, name='tabu_tenure'),
        Categorical(['random', 'greedy', 'kmeans'], name='inicializacion')
    ]
    
    # Función objetivo con decorador para argumentos nombrados
    @use_named_args(dimensions)
    def objective(**params_dict):
        return evaluate_tabu_hyperparams(
            tabu_tenure=params_dict['tabu_tenure'],
            inicializacion=params_dict['inicializacion'],
            params=params,
            instance=instance,
            result_dir=result_dir,
            logger=logger,
            problem=problem,
            num_runs=num_runs,
            max_iter=max_iter,
            num_processes=num_processes
        )
    
    start_time = time.time()
    
    # Ejecutar optimización bayesiana
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        acq_func='EI',  # Expected Improvement
        random_state=random_state,
        n_jobs=1  # Paralelización se maneja internamente
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Procesar resultados
    all_results = []
    for i, (x_val, y_val) in enumerate(zip(result.x_iters, result.func_vals)):
        all_results.append({
            "instance": instance+1,
            "iteration": i+1,
            "tabu_tenure": int(x_val[0]),
            "inicializacion": x_val[1],
            "best_cost": y_val,
            "is_best": i == np.argmin(result.func_vals)
        })
    
    # Obtener mejor resultado
    best_tabu_tenure, best_inicializacion = result.x
    best_tabu_tenure = int(best_tabu_tenure)
    best_cost = result.fun
    
    logger.info(f"\n=== Resultado de Búsqueda Bayesiana TABU ===")
    logger.info(f"Mejor combinación encontrada:")
    logger.info(f"  Tabu tenure: {best_tabu_tenure}")
    logger.info(f"  Inicialización: {best_inicializacion}")
    logger.info(f"  Mejor costo: {best_cost:.2f}")
    logger.info(f"Tiempo total: {total_time:.2f} segundos")
    
    # Calcular cuántas combinaciones habría tenido el grid search
    num_clientes = len(params['I'])
    grid_combinations = 5 * 3  # tabu_tenure options * inicializacion options
    logger.info(f"Mejora estimada vs grid search: {((grid_combinations*num_runs) - n_calls*num_runs) / (grid_combinations*num_runs):.1%} menos evaluaciones")
    
    # Guardar resultados
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(result_dir, f"tabu_bayesian_results_instance{instance+1}.csv")
    df.to_csv(csv_path, index=False)
    
    # Guardar también el historial de optimización
    history_path = os.path.join(result_dir, f"tabu_bayesian_history_instance{instance+1}.csv")
    history_df = pd.DataFrame({
        'iteration': range(1, len(result.func_vals)+1),
        'tabu_tenure': [int(x[0]) for x in result.x_iters],
        'inicializacion': [x[1] for x in result.x_iters],
        'cost': result.func_vals,
        'cumulative_best': np.minimum.accumulate(result.func_vals)
    })
    history_df.to_csv(history_path, index=False)
    
    logger.info(f"Resultados guardados en: {csv_path}")
    logger.info(f"Historial guardado en: {history_path}")
    
    return df

# Mantener las funciones originales para compatibilidad
def run_grasp_once(params, instance, result_dir, alpha, frac_neighbors, logger, problem, max_iter):
    grasp = GRASPSearch(params=params, instance=instance, result_dir=result_dir, alpha=alpha, frac_neighbors=frac_neighbors, logger=logger, problem=problem, max_iter=max_iter)
    grasp.run()
    return deepcopy(grasp.best_solution)

def run_grasp_committee(params, instance, result_dir, logger, alpha=0.3, frac_neighbors=4, problem="P2", max_iter=None, num_runs=5):
    # Redirigir a la versión paralela
    return run_grasp_committee_parallel(params, instance, result_dir, logger, alpha, frac_neighbors, problem, max_iter, num_runs)

def run_grasp_hyperparam_search(params, instance, result_dir, logger, problem="P2", num_runs=8, max_iter=None, type_search="grid"):
    # Redirigir a la versión paralela
    if type_search=="grid":
        return run_grasp_hyperparam_grid_search(params, instance, result_dir, logger, problem, num_runs, max_iter)
    elif type_search=="random":
        return run_grasp_hyperparam_random_search(params, instance, result_dir, logger, problem, num_runs, max_iter)
    elif type_search=="bayesian":
        return run_grasp_hyperparam_bayesian_search(params, instance, result_dir, logger, problem, num_runs, max_iter)

def run_tabu_once(params, instance, result_dir, problem, logger, inicialization, tabu_tenure, max_iter):
    tabu = TabuSearch(params=params, inicialization=inicialization, instance=instance, problem=problem, result_dir=result_dir, tabu_tenure=tabu_tenure, max_iter=max_iter, logger=logger)
    tabu.run()
    return deepcopy(tabu.best_solution)

def run_tabu_committee(params, instance, result_dir, logger, inicialization="random", max_iter=50, num_runs=5, tabu_tenure=7, problem="P2"):
    # Redirigir a la versión paralela
    return run_tabu_committee_parallel(params, instance, result_dir, logger, inicialization, max_iter, num_runs, tabu_tenure, problem)

def run_tabu_hyperparam_search(params, instance, result_dir, logger, problem="P2", num_runs=8, max_iter=None, type_search="grid"):
    if type_search=="grid":
        return run_tabu_hyperparam_grid_search(params=params, instance=instance, result_dir=result_dir, logger=logger, problem=problem, num_runs=num_runs, max_iter=max_iter)
    elif type_search=="random":
        return run_tabu_hyperparam_random_search(params=params, instance=instance, result_dir=result_dir, logger=logger, problem=problem, num_runs=num_runs, max_iter=max_iter)
    elif type_search=="bayesian":
        return run_tabu_hyperparam_bayesian_search(params=params, instance=instance, result_dir=result_dir, logger=logger, problem=problem, num_runs=num_runs, max_iter=max_iter)

def run_genetic_committee(params, instance, result_dir, problem, inicializacion, logger, generations: int = 50, mutation_rate: float = 0.05, crossover_rate: float = 0.95, tournament: int = 5):
    # Redirigir a la versión paralela
    return run_genetic_committee_parallel(params=params, instance=instance, problem=problem, result_dir=result_dir, inicializacion=inicializacion, generations=generations, mutation_rate=mutation_rate, crossover_rate=crossover_rate, tournament=tournament, logger=logger)

def run_genetic_once(params, instance, result_dir, problem, inicializacion, logger, generations: int = 50, mutation_rate: float = 0.05, crossover_rate: float = 0.95, tournament: int = 5):
    try:    
        genetic = GeneticSearch(params=params, instance=instance, problem=problem, result_dir=result_dir, inicializacion=inicializacion, generations=generations, mutation_rate=mutation_rate, crossover_rate=crossover_rate, tournament=tournament, logger=logger)
        genetic.run()
    except Exception as e:
        logger.error(f"Error procesando instancia {instance}: {e}")
        logger.error(traceback.format_exc()) 
    return deepcopy(genetic.best_individual)

def save_solution(sol, time, costs, alpha=None, num_neighbors=None, instance=None, result_dir=None):
    """Función save_solution actualizada para manejar tanto GRASP como TABU"""
    os.makedirs(result_dir, exist_ok=True)
    filepath = os.path.join(result_dir, f"{instance+1}_best_committee_solution.txt")
    
    with open(filepath, 'w') as f:
        f.write(f"Instancia: {instance + 1}\n")
        if num_neighbors is not None:
            f.write(f"Número de vecinos: {num_neighbors}\n")
        if alpha is not None:
            f.write(f"Alfa: {alpha}\n")
        f.write(f"Costo mejor solución (comité): {sol.cost:.2f}\n")
        f.write(f"Tiempo mejor solución (comité): {sol.time:.2f} segundos\n")
        f.write(f"Instalaciones abiertas:\n")
        for j, val in sol.y.items():
            if val == 1:
                f.write(f"  Instalación {j}\n")
        f.write(f"Puntos de recogida abiertos:\n")
        for k, val in sol.nu.items():
            if val == 1:
                f.write(f"  Punto {k}\n")
        f.write("Detalles del comité:\n")
        f.write(f"Miembros del comité: {len(costs)}\n")
        f.write(f"Tiempo total del comité: {time:.2f} segundos\n")
        f.write(f"Media de costos del comité: {np.mean(costs):.2f}\n")
        f.write(f"Desviación estándar de costos del comité: {np.std(costs):.2f}\n")

