import json
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import time
import logging
import traceback
from copy import deepcopy
from itertools import product
import random

# Tus imports existentes
from algorithms.GRASP import GRASPSearch
from algorithms.GeneticAlgorithm import GeneticSearch
from algorithms.TABU import TabuSearch
from structure.create_instances import create_params

random.seed(42)

def load_optimal_solutions():
    """Carga las soluciones óptimas desde los archivos JSON"""
    optimal_solutions = {}
    
    for n in ['n_10', 'n_50', 'n_100']:
        json_path = f"data/optimal_solutions/{n}.json"
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                optimal_solutions.update(data)
    
    return optimal_solutions

def get_combo_columns(algorithm_type):

    # Definir parámetros relevantes según el algoritmo
    algo_params = {
        'tabu': ['tabu_tenure', 'inicializacion', 'time_limit'],
        'genetic': ['mutation_rate', 'tournament', 'inicializacion'],
        'grasp': ['alpha', 'frac_neighbors', 'max_iter'],
        # puedes añadir más algoritmos aquí...
    }
    params = algo_params.get(algorithm_type.lower(), [])
    return params

def calculate_gap(primal_cost, optimal_cost):
    """
    Calcula el GAP de optimalidad según la fórmula:
    GAP(%) = 100 * (z_primal - z_dual) / z_primal
    """
    if optimal_cost is None or primal_cost <= 0:
        return None
    # print(primal_cost)
    # print("OPTIMAL ", optimal_cost)
    gap = abs((primal_cost - optimal_cost) / primal_cost)
    return max(0, gap)  # GAP no puede ser negativo

def analyze_results(costs, optimal_cost=None):
    """Analiza los resultados de múltiples ejecuciones"""
    costs_array = np.array(costs)
    # print(float(np.std(costs_array)))
    # print(optimal_cost)
    
    analysis = {
        'mean_cost': float(np.mean(costs_array)),
        'std_cost': float(np.std(costs_array)),
        'best_cost': float(np.min(costs_array)),
        'worst_cost': float(np.max(costs_array)),
        'num_runs': len(costs),
        'optimal_solutions': 0,
        'non_optimal_solutions': len(costs)
    }
    
    if optimal_cost is not None:
        optimal_cost = float(optimal_cost)
        # Calcular GAPs
        gaps = [calculate_gap(cost, optimal_cost) for cost in costs]
        # print("GAPS ", gaps)
        valid_gaps = [g for g in gaps if g is not None]
        # print("VALID GAP ", valid_gaps)
        if valid_gaps:
            analysis['mean_gap'] = float(np.mean(valid_gaps))
            analysis['best_gap'] = float(np.min(valid_gaps))
            analysis['std_gap'] = float(np.std(valid_gaps))
        
        # Contar soluciones óptimas (con tolerancia de 1e-6)
        tolerance = 1e-6
        optimal_count = sum(1 for cost in costs if abs(cost - optimal_cost) <= tolerance)
        analysis['optimal_solutions'] = optimal_count
        analysis['non_optimal_solutions'] = len(costs) - optimal_count
    
    return analysis

def summarize_results(df_all, optimal_solutions):
    """
    A partir de los resultados crudos (df_all), calcular métricas agregadas por combinación.
    """
    summary = []

    for (combo_idx, instance_name), group in df_all.groupby(['combo_idx', 'instance_name']):
        costs = group['cost'].tolist()
        times = group['time'].tolist()

        optimal_cost = optimal_solutions.get(instance_name)
        analysis = analyze_results(costs, optimal_cost)

        entry = {
            'combo_idx': combo_idx,
            'instance_name': instance_name,
            **analysis,
            'mean_time': float(np.mean(times))
        }

        # Añadir los parámetros del combo (ejemplo: mutation_rate, tournament...)
        for col in ['mutation_rate', 'tournament', 'inicializacion', 'tabu_tenure', 'max_iter', 'gamma_f', 'gamma_q']:
            if col in group.columns:
                entry[col] = group[col].iloc[0]

        summary.append(entry)

    return pd.DataFrame(summary)


def save_compact_results(results_data, result_dir, algorithm_name, search_type="committee"):
    """Guarda los resultados de forma compacta y clara"""
    
    # Crear directorio principal
    os.makedirs(result_dir, exist_ok=True)
    
    if search_type == "committee":
        # Para comités, solo guardar un resumen por instancia
        summary_file = os.path.join(result_dir, f"{algorithm_name}_committee_summary.csv")
        
        df = pd.DataFrame(results_data)
        df = df.round(4)  # Redondear a 4 decimales
        df.to_csv(summary_file, index=False)
        
        # Guardar también un archivo detallado con estadísticas
        detailed_file = os.path.join(result_dir, f"{algorithm_name}_committee_detailed.json")
        with open(detailed_file, 'w') as f:
            json.dump(results_data, f, indent=2)
            
    elif search_type == "hyperparameters":
        # Para hiperparámetros, estructura más compacta
        
        # Resumen general
        summary_file = os.path.join(result_dir, f"{algorithm_name}_hyperparams_summary.csv")
        
        # Convertir a DataFrame y ordenar por mejor costo
        df = pd.DataFrame(results_data)
        df = df.sort_values('best_cost')
        df = df.round(4)
        df.to_csv(summary_file, index=False)
        
        # Top 10 mejores combinaciones
        top10_file = os.path.join(result_dir, f"{algorithm_name}_top10_hyperparams.csv")
        df.head(10).to_csv(top10_file, index=False)
        
        # Estadísticas por instancia si hay múltiples instancias
        if 'instance' in df.columns:
            # Crea un diccionario para las agregaciones
            agg_dict = {
                'best_cost': ['mean', 'min', 'max'],
                'mean_cost': 'mean',
            }
            
            # Añade 'mean_gap' solo si existe en el DataFrame
            if 'mean_gap' in df.columns:
                agg_dict['mean_gap'] = 'mean'
            
            stats_by_instance = df.groupby('instance').agg(agg_dict).round(4)
            
            stats_file = os.path.join(result_dir, f"{algorithm_name}_stats_by_instance.csv")
            stats_by_instance.to_csv(stats_file)

def save_global_results(all_results, result_dir, algorithm_name):
    os.makedirs(result_dir, exist_ok=True)

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(result_dir, f"{algorithm_name}_all_results.csv"), index=False)

    # También un Excel si quieres explorar mejor
    # df.to_excel(os.path.join(result_dir, f"{algorithm_name}_all_results.xlsx"), index=False)
    return df


def run_single_algorithm_execution(args):
    """Función genérica para ejecutar una sola ejecución de cualquier algoritmo"""
    algorithm_type = args['algorithm_type']
    
    if algorithm_type == 'GRASP':
        return run_single_grasp_execution_improved(args)
    elif algorithm_type == 'GENETIC':
        return run_single_genetic_execution_improved(args)
    elif algorithm_type == 'TABU':
        return run_single_tabu_execution_improved(args)
    else:
        raise ValueError(f"Tipo de algoritmo desconocido: {algorithm_type}")

def run_single_grasp_execution_improved(args):
    """Ejecuta una sola ejecución de GRASP con logging mejorado"""
    params = args['params']
    instance = args['instance']
    instance_name = args['instance_name']
    n_nodes = args['n_nodes']
    alpha = args['alpha']
    frac_neighbors = args['frac_neighbors']
    problem = args['problem']
    max_iter = args['max_iter']
    run_idx = args['run_idx']
    temp_dir = args.get('temp_dir', '/tmp')
    
    # Crear directorio temporal para esta ejecución
    run_dir = os.path.join(temp_dir, f"grasp_run_{run_idx}_{os.getpid()}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Logger simplificado
    logger = logging.getLogger(f'grasp_worker_{os.getpid()}_{run_idx}')
    logger.setLevel(logging.WARNING)  # Reducir logging para evitar spam
    
    try:
        grasp = GRASPSearch(
            params=params,
            instance=instance_name,
            result_dir=run_dir,
            alpha=alpha,
            frac_neighbors=frac_neighbors,
            logger=logger,
            problem=problem,
            max_iter=max_iter
        )
        start_time = time.time()
        grasp.run()
        print(f"Ha tardado {time.time()-start_time}")
        return {
            'algorithm_type': 'GRASP',
            'instance': instance,
            'instance_name': instance_name,
            'n_nodes': n_nodes,
            'run_idx': run_idx,
            'cost': round(grasp.best_solution.cost, 2),
            'time': grasp.best_solution.time,
            # 'solution': deepcopy(grasp.best_solution),
            'max_iter': max_iter,
            'alpha': alpha,
            'frac_neighbors': frac_neighbors,
            'success': True
        }
    except Exception as e:
        logger.error(f"Error en GRASP run {run_idx}: {e}")
        return {
            'algorithm_type': 'GRASP',
            'instance': instance,
            'instance_name': instance_name,
            'n_nodes': n_nodes,
            'run_idx': run_idx,
            'cost': float('inf'),
            'time': 0,
            # 'solution': None,
            'max_iter': max_iter,
            'alpha': alpha,
            'frac_neighbors': frac_neighbors,
            'success': False,
            'error': str(e)
        }
    finally:
        # Limpiar directorio temporal
        try:
            import shutil
            shutil.rmtree(run_dir, ignore_errors=True)
        except:
            pass

def run_single_genetic_execution_improved(args):
    """Ejecuta una sola ejecución de Genetic Algorithm con logging mejorado"""
    params = args['params']
    instance = args['instance']
    instance_name = args['instance_name']
    n_nodes = args['n_nodes']
    combo_idx = args['combo_idx']
    # generations = args['generations']
    mutation_rate = args['mutation_rate']
    # crossover_rate = args['crossover_rate']
    tournament = args['tournament']
    inicializacion = args['inicializacion']
    problem = args['problem']
    run_idx = args['run_idx']
    temp_dir = args.get('temp_dir', '/tmp')
    
    run_dir = os.path.join(temp_dir, f"genetic_run_{run_idx}_{os.getpid()}")
    os.makedirs(run_dir, exist_ok=True)
    
    logger = logging.getLogger(f'genetic_worker_{os.getpid()}_{run_idx}')
    logger.setLevel(logging.WARNING)
    # print("Inicialización ", inicializacion)
    # print("Mutation rate ", mutation_rate)
    # print("Tournament ", tournament)
    try:
        genetic = GeneticSearch(
            params=params,
            instance=instance,
            problem=problem,
            result_dir=run_dir,
            inicializacion=inicializacion,
            # generations=generations,
            mutation_rate=mutation_rate,
            # crossover_rate=crossover_rate,
            tournament=tournament,
            logger=logger
        )
        start_time = time.time()
        genetic.run()
        print(f"Ha tardado {time.time()-start_time}")
        
        return {
            'algorithm_type': 'GENETIC',
            'instance': instance,
            'instance_name': instance_name,
            'n_nodes': n_nodes,
            'combo_idx': combo_idx,
            'run_idx': run_idx,
            'cost': round(genetic.best_individual.cost, 2),
            'time': genetic.best_individual.time,
            # 'solution': deepcopy(genetic.best_individual),
            # 'generations': generations,
            'mutation_rate': mutation_rate,
            # 'crossover_rate': crossover_rate,
            'tournament': tournament,
            'inicializacion': inicializacion,
            'success': True
        }
    except Exception as e:
        logger.error(f"Error en Genetic run {run_idx}: {e}")
        return {
            'algorithm_type': 'GENETIC',
            'instance': instance,
            'instance_name': instance_name,
            'n_nodes': n_nodes,
            'combo_idx': combo_idx,
            'run_idx': run_idx,
            'cost': float('inf'),
            'time': 0,
            # 'solution': None,
            # 'generations': generations,
            'mutation_rate': mutation_rate,
            # 'crossover_rate': crossover_rate,
            'tournament': tournament,
            'inicializacion': inicializacion,
            'success': False,
            'error': str(e)
        }
    finally:
        try:
            import shutil
            shutil.rmtree(run_dir, ignore_errors=True)
        except:
            pass

def run_single_tabu_execution_improved(args):
    """Ejecuta una sola ejecución de Tabu Search con logging mejorado"""
    params = args['params']
    instance = args['instance']
    instance_name = args['instance_name']
    n_nodes = args['n_nodes']
    time_limit = args['time_limit']
    tabu_tenure = args['tabu_tenure']
    inicializacion = args['inicializacion']
    # max_iter = args['max_iter']
    # gamma_f = args['gamma_f']
    # gamma_q = args['gamma_q']
    time_limit = args['time_limit']
    problem = args['problem']
    run_idx = args['run_idx']
    temp_dir = args.get('temp_dir', '/tmp')
    
    run_dir = os.path.join(temp_dir, f"tabu_run_{run_idx}_{os.getpid()}")
    os.makedirs(run_dir, exist_ok=True)
    
    logger = logging.getLogger(f'tabu_worker_{os.getpid()}_{run_idx}')
    logger.setLevel(logging.DEBUG)
    
    try:
        tabu = TabuSearch(
            params=params,
            inicializacion=inicializacion,
            instance=instance,
            problem=problem,
            result_dir=run_dir,
            tabu_tenure=tabu_tenure,
            # max_iter_without_improvement=max_iter,
            # gamma_f=gamma_f,
            # gamma_q=gamma_q,
            time_limit=time_limit,
            logger=logger
        )
        start_time = time.time()
        tabu.run()
        print(f"Ha tardado {time.time()-start_time}")
        
        return {
            'algorithm_type': 'tabu',
            'instance': instance,
            'instance_name': instance_name,
            'n_nodes': n_nodes,
            'run_idx': run_idx,
            'cost': round(tabu.best_solution.cost, 2),
            'time': tabu.best_solution.time,
            # 'solution': deepcopy(tabu.best_solution),
            'tabu_tenure': tabu_tenure,
            'inicializacion': inicializacion,
            # 'max_iter': max_iter,
            # 'gamma_f': gamma_f,
            # 'gamma_q': gamma_q,
            'time_limit': time_limit,
            'success': True
        }
    except Exception as e:
        logger.error(f"Error en Tabu run {run_idx}: {e}")
        return {
            'algorithm_type': 'TABU',
            'instance': instance,
            'instance_name': instance_name,
            'n_nodes': n_nodes,
            'run_idx': run_idx,
            'cost': float('inf'),
            'time': 0,
            # 'solution': None,
            'tabu_tenure': tabu_tenure,
            'inicializacion': inicializacion,
            # 'max_iter': max_iter,
            # 'gamma_f': gamma_f,
            # 'gamma_q': gamma_q,
            'time_limit': time_limit,
            'success': False,
            'error': str(e)
        }
    finally:
        try:
            import shutil
            shutil.rmtree(run_dir, ignore_errors=True)
        except:
            pass

def run_committee_multiple_instances(algorithm_params_list, instances_data, n_nodes, result_base_dir, 
                                  algorithm_name, num_runs=5, num_processes=None):
    """
    Ejecuta comités para múltiples instancias de forma eficiente
    
    Args:
        algorithm_params_list: Lista de diccionarios con parámetros por instancia
        instances_data: Lista de datos de instancias
        result_base_dir: Directorio base para resultados
        algorithm_name: Nombre del algoritmo
        num_runs: Número de ejecuciones por instancia
        num_processes: Número de procesos paralelos
    """
    
    if num_processes is None:
        num_processes = min(cpu_count(), len(algorithm_params_list) * num_runs)
    
    optimal_solutions = load_optimal_solutions()
    
    print(f"\n=== Ejecutando comités {algorithm_name.upper()} ===")
    print(f"Instancias: {len(instances_data)}, Ejecuciones por instancia: {num_runs}")
    print(f"Procesos paralelos: {num_processes}")
    
    start_time = time.time()
    
    # Preparar todos los argumentos
    all_args = []
    for i, (instance_params, instance_data) in enumerate(zip(algorithm_params_list, instances_data)):
        instance_name = instance_data[0]
        
        for run_idx in range(num_runs):
            args = {
                **instance_params,
                'instance': i,
                'instance_name': instance_name,
                'n_nodes': n_nodes,
                'run_idx': run_idx,
                'temp_dir': os.path.join('/tmp', f'{algorithm_name}_committee')
            }
            all_args.append(args)
    
    # Ejecutar todas las ejecuciones en paralelo
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        all_results = list(executor.map(run_single_algorithm_execution, all_args))
    
    # Procesar resultados por instancia
    results_by_instance = {}
    for result in all_results:
        instance_idx = result['instance']
        if instance_idx not in results_by_instance:
            results_by_instance[instance_idx] = []
        results_by_instance[instance_idx].append(result)
    
    # Analizar y guardar resultados
    summary_data = []
    
    for instance_idx, instance_results in results_by_instance.items():
        instance_name = instance_data[0]
        
        # Filtrar solo resultados exitosos
        successful_results = [r for r in instance_results if r['success']]
        
        if not successful_results:
            print(f"WARNING: No hay resultados exitosos para instancia {instance_name}")
            continue
        
        costs = [r['cost'] for r in successful_results]
        times = [r['time'] for r in successful_results]
        
        # Buscar solución óptima
        optimal_cost = optimal_solutions.get(instance_name)
        
        # Análisis completo
        analysis = analyze_results(costs, optimal_cost)
        
        # Datos del resumen
        instance_summary = {
            'instance': instance_idx + 1,
            'instance_name': instance_name,
            'n_nodes': n_nodes,
            'algorithm': algorithm_name,
            **analysis,
            'mean_time': float(np.mean(times)),
            'best_time': float(np.min(times)),
            'total_time': float(np.sum(times)),
            'optimal_value': optimal_cost,
            'successful_runs': len(successful_results),
            'failed_runs': len(instance_results) - len(successful_results)
        }
        
        # Añadir parámetros específicos del algoritmo
        if algorithm_name == 'GRASP':
            first_result = successful_results[0]
            instance_summary.update({
                'alpha': first_result['alpha'],
                'frac_neighbors': first_result['frac_neighbors']
            })
        elif algorithm_name == 'GENETIC':
            first_result = successful_results[0]
            instance_summary.update({
                'generations': first_result['generations'],
                'mutation_rate': first_result['mutation_rate'],
                'crossover_rate': first_result['crossover_rate'],
                'tournament': first_result['tournament'],
                'inicializacion': first_result['inicializacion']
            })
        elif algorithm_name == 'TABU':
            first_result = successful_results[0]
            instance_summary.update({
                'tabu_tenure': first_result['tabu_tenure'],
                'inicializacion': first_result['inicializacion']
            })
        
        summary_data.append(instance_summary)
        
        # Log de progreso
        gap_str = f", GAP: {analysis.get('best_gap', 'N/A'):.2f}%" if optimal_cost else ""
        print(f"Instancia {instance_name}: Mejor costo: {analysis['best_cost']:.2f}{gap_str}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Guardar resultados compactos
    result_dir = os.path.join(result_base_dir, f"{algorithm_name}_committee_results")
    save_compact_results(summary_data, result_dir, algorithm_name, "committee")

    # Guardar todos los resultados crudos (ccada combinación)
    df_all = save_global_results(all_results, result_dir, algorithm_name)

    # Agrupar por nº de nodos
    stats_by_nnodes = df_all.groupby(['n_nodes', 'algorithm_type']).agg({
        'cost': ['mean', 'std', 'min', 'max'],
        'time': ['mean', 'sum'],
    }).round(4)
    stats_by_nnodes.to_csv(os.path.join(result_dir, f"{algorithm_name}_global_stats_by_nnodes.csv"), mode='a', header=False)

    
    print(f"\n=== Comités {algorithm_name.upper()} completados ===")
    print(f"Tiempo total: {total_time:.2f} segundos")
    print(f"Promedio por instancia: {total_time/len(instances_data):.2f} segundos")
    print(f"Resultados guardados en: {result_dir}")
    
    return summary_data

def run_hyperparameter_search_multiple_instances(algorithm_name, param_combinations, instances_data, n_nodes,
                                               result_base_dir, num_runs=8, num_processes=None):
    """
    Búsqueda de hiperparámetros para múltiples instancias con estructura compacta
    """
    
    total_executions = len(param_combinations) * len(instances_data) * num_runs
    
    if num_processes is None:
        num_processes = min(cpu_count(), total_executions // 4)  # Ser conservador con la memoria
    
    optimal_solutions = load_optimal_solutions()
    
    print(f"\n=== Búsqueda de hiperparámetros {algorithm_name.upper()} ===")
    print(f"Combinaciones: {len(param_combinations)}, Instancias: {len(instances_data)}")
    print(f"Ejecuciones por combinación: {num_runs}, Total: {total_executions}")
    print(f"Procesos paralelos: {num_processes}")
    
    start_time = time.time()
    
    # Preparar todas las ejecuciones
    all_args = []
    
    for combo_idx, param_combo in enumerate(param_combinations):
        for instance_idx, instance_data in enumerate(instances_data):
            instance_name = instance_data[0]
            params = create_params(instance=instance_idx, path=instance_data[1])
            for run_idx in range(num_runs):
                args = {
                    'algorithm_type': algorithm_name,
                    'params': params,  # Datos de la instancia
                    'instance': instance_idx,
                    'instance_name': instance_name,
                    'n_nodes': n_nodes,
                    'run_idx': run_idx,
                    'combo_idx': combo_idx,
                    'temp_dir': os.path.join('/tmp', f'{algorithm_name}_hyperparam'),
                    **param_combo  # Parámetros específicos del algoritmo
                }
                all_args.append(args)
    
    # Ejecutar en lotes para manejar memoria
    batch_size = max(100, num_processes * 2)
    all_results = []
    
    for i in range(0, len(all_args), batch_size):
        batch_args = all_args[i:i + batch_size]
        print(f"Procesando lote {i//batch_size + 1}/{(len(all_args)-1)//batch_size + 1}")
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            batch_results = list(executor.map(run_single_algorithm_execution, batch_args))
        
        all_results.extend(batch_results)
    
    # Procesar resultados
    results_by_combo_instance = {}
    
    for result in all_results:
        instance_name = result['instance_name']
        optimal_cost = optimal_solutions.get(instance_name)
        result['optimal_cost'] = round(float(optimal_cost), 2)
        
        if result['optimal_cost'] is not None:
            result['gap'] = calculate_gap(result['cost'], result['optimal_cost'])
            # print("ABS ", abs(result['cost'] - float(optimal_cost)))
            result['is_optimal'] = int(abs(result['cost'] - result['optimal_cost']) <= 1e-6)
        else:
            result['gap'] = None
            result['is_optimal'] = 0
        key = (result.get('combo_idx', 0), result['instance'])
        if key not in results_by_combo_instance:
            results_by_combo_instance[key] = []
        results_by_combo_instance[key].append(result)
    
    # Analizar cada combinación para cada instancia
    hyperparameter_summary = []
    
    for combo_idx, param_combo in enumerate(param_combinations):
        for instance_idx, instance_data in enumerate(instances_data):
            instance_name = instance_data[0]
            key = (combo_idx, instance_idx)
            
            if key not in results_by_combo_instance:
                continue
                
            combo_results = results_by_combo_instance[key]
            successful_results = [r for r in combo_results if r['success']]
            
            if not successful_results:
                continue
            
            costs = [r['cost'] for r in successful_results]
            times = [r['time'] for r in successful_results]
            
            optimal_cost = optimal_solutions.get(instance_name)
            optimal_cost = round(float(optimal_cost), 2)
            analysis = analyze_results(costs, optimal_cost)
            # print(analysis)
            
            # Crear entrada del resumen
            combo_summary = {
                'combo_idx': combo_idx,
                'instance': instance_idx + 1,
                'instance_name': instance_name,
                'n_nodes': n_nodes,
                'algorithm': algorithm_name,
                **param_combo,  # Parámetros de la combinación
                **analysis,
                'mean_time': float(np.mean(times)),
                'optimal_value': optimal_cost
            }
            
            hyperparameter_summary.append(combo_summary)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Guardar resultados compactos
    result_dir = os.path.join(result_base_dir, f"{algorithm_name}_hyperparameter_search")
    save_compact_results(hyperparameter_summary, result_dir, algorithm_name, "hyperparameters")

    # Guardar todos los resultados crudos (ccada combinación)
    df_all = save_global_results(all_results, result_dir, algorithm_name)
    # df_all = add_combo_index(df_all, algorithm_name)

    # summary_df = summarize_results(df_all, optimal_solutions)

    # generate_final_report(summary_df.to_dict(orient='records'), result_dir, algorithm_name, "hyperparameters")
    columns_names = get_combo_columns(algorithm_type=algorithm_name)
    # Agrupar por nº de nodos
    columns_names.append('n_nodes')
    print(df_all.columns)
    stats_by_nnodes = df_all.groupby(columns_names).agg({
        'cost': ['mean', 'std', 'min', 'max'],
        'time': ['mean', 'sum'],
        'gap': 'mean',                 # GAP promedio
        'is_optimal': 'sum'            # Nº de soluciones óptimas
    }).round(4)
    stats_by_nnodes.to_csv(os.path.join(result_dir, f"{algorithm_name}_global_stats_by_nnodes.csv"))
    
    # Estadísticas finales
    print(f"\n=== Búsqueda de hiperparámetros {algorithm_name.upper()} completada ===")
    print(f"Tiempo total: {total_time:.2f} segundos")
    print(f"Tiempo promedio por combinación: {total_time/len(param_combinations):.2f} segundos")
    print(f"Resultados guardados en: {result_dir}")
    
    return stats_by_nnodes

# Funciones específicas para cada algoritmo
def generate_grasp_param_combinations(search_type="grid", num_combinations=None):
    """Genera combinaciones de parámetros para GRASP"""
    if search_type == "grid":
        alphas = [0.1, 0.3, 0.5, 0.7]
        frac_neighbors_list = [2, 4, 6]
        max_iter = [20]
        combinations = list(product(alphas, frac_neighbors_list, max_iter))
        
        return [{'algorithm_type': 'GRASP', 'alpha': a, 'frac_neighbors': fn, 'problem': 'P2', 'max_iter': mi} 
                for a, fn, mi in combinations]
    
    elif search_type == "random":
        if num_combinations is None:
            num_combinations = 5
        
        combinations = []
        for _ in range(num_combinations):
            alpha = round(random.uniform(0.05, 0.95), 3)
            frac_neighbors = random.randint(2, 6)
            max_iter = random.choice([10, 25, 50])
            combinations.append({
                'algorithm_type': 'GRASP',
                'alpha': alpha,
                'frac_neighbors': frac_neighbors,
                'problem': 'P2',
                'max_iter': max_iter
            })
        return combinations
    
    else:
        raise ValueError(f"Tipo de búsqueda desconocido: {search_type}")

def generate_genetic_param_combinations(search_type="grid", num_combinations=None):
    """Genera combinaciones de parámetros para Genetic Algorithm"""
    if search_type == "grid":
        # generations_list = [50, 75]
        mutation_rates = [0.05, 0.1]
        # crossover_rates = [0.9, 0.95, 0.99]
        tournaments = [3, 4, 5]
        inicializaciones = ["random", "greedy"]
        
        combinations = list(product(tournaments, mutation_rates, inicializaciones))
        
        return [{'algorithm_type': 'GENETIC', 'mutation_rate': mr, 
                'tournament': t, 'inicializacion': init, 'problem': 'P2'} 
                for t, mr, init in combinations]
    
    elif search_type == "random":
        if num_combinations is None:
            num_combinations = 5
        
        combinations = []
        for _ in range(num_combinations):
            # generations = random.choice([25, 50, 75])
            mutation_rate = round(random.uniform(0.01, 0.1), 3)
            # crossover_rate = round(random.uniform(0.9, 0.99), 3)
            tournament = random.randint(3, 10)
            inicializacion = random.choice(["random", "greedy"])
            
            combinations.append({
                'algorithm_type': 'GENETIC',
                # 'generations': generations,
                'mutation_rate': mutation_rate,
                # 'crossover_rate': crossover_rate,
                'tournament': tournament,
                'inicializacion': inicializacion,
                'problem': 'P2'
            })
        return combinations
    
    else:
        raise ValueError(f"Tipo de búsqueda desconocido: {search_type}")

def generate_tabu_param_combinations(search_type="grid", num_combinations=None):
    """Genera combinaciones de parámetros para Tabu Search"""
    if search_type == "grid":
        tabu_tenures = [0.25, 0.5]
        inicializaciones = ["random", "greedy", "kmeans"]
        time_limit = [50, 120]
        # max_iters = [3, 5, 10]
        # gamma_fs = [0.2, 0.5, 0.8]
        # gamma_qs = [0.2, 0.5, 0.8]
        
        combinations = list(product(tabu_tenures, inicializaciones, time_limit))
        
        return [{'algorithm_type': 'TABU', 'tabu_tenure': tt, 'inicializacion': init,
                 'time_limit': time_limit, 'problem': 'P2'} 
                for tt, init, time_limit in combinations]
    
    elif search_type == "random":
        if num_combinations is None:
            num_combinations = 5
        
        combinations = []
        for _ in range(num_combinations):
            tabu_tenure = round(random.uniform(0.05, 0.6), 3)
            inicializacion = random.choice(["random", "greedy", "kmeans"])
            time_limit = random.choice([10, 50, 100])
            # max_iter = random.randint(3, 10)
            # gamma_f = round(random.uniform(0.5, 1.0), 3)
            # gamma_q = round(random.uniform(0.5, 1.0), 3)
            
            combinations.append({
                'algorithm_type': 'TABU',
                'tabu_tenure': tabu_tenure,
                'inicializacion': inicializacion,
                # 'max_iter': max_iter,
                # 'gamma_f': gamma_f,
                # 'gamma_q': gamma_q,
                'time_limit': time_limit,
                'problem': 'P2'
            })
        return combinations
    
    else:
        raise ValueError(f"Tipo de búsqueda desconocido: {search_type}")
    
def preprocess_grouped_df(df):
    """
    Convierte un DataFrame con MultiIndex en columnas (resultado de groupby + agg)
    en uno plano con nombres esperados por generate_final_report.
    """
    # Aplanar MultiIndex
    df = df.copy()
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

    # Renombrar columnas clave
    rename_map = {
        'cost_min': 'best_cost',
        'cost_mean': 'mean_cost',
        'cost_std': 'std_cost',
        'gap_mean': 'best_gap',   # si solo tienes media de gap
        'is_optimal_sum': 'optimal_solutions',
        'time_mean': 'mean_time',
        'time_sum': 'total_time'
    }
    df = df.rename(columns=rename_map)

    return df.reset_index()


def generate_final_report(df, result_dir, algorithm, mode): 
    """Genera un reporte final consolidado con las mejores métricas""" 
    if df.empty: 
        return 
    
    df = preprocess_grouped_df(df)
    # print(df.columns)
    
    # Reporte general 
    report_lines = [ 
        f"=== REPORTE FINAL - {algorithm.upper()} ({mode.upper()}) ===\n", 
        f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}", 
        f"Total de configuraciones: {len(df)}"
        "" 
    ] 
    if mode == 'committee': 
        # Estadísticas de comités 
        if 'best_cost' in df.columns: 
            report_lines.extend([ 
                "=== ESTADÍSTICAS GENERALES ===", 
                f"Mejor costo global: {df['best_cost'].min():.4f}", 
                f"Costo promedio: {df['best_cost'].mean():.4f}", 
                f"Desviación estándar: {df['best_cost'].std():.4f}", 
                "" 
            ]) 
        if 'best_gap' in df.columns and df['best_gap'].notna().any(): 
            valid_gaps = df[df['best_gap'].notna()]['best_gap'] 
            report_lines.extend([ 
                "=== ANÁLISIS DE GAP ===", 
                f"Mejor GAP: {valid_gaps.min():.4f}%", 
                f"GAP promedio: {valid_gaps.mean():.4f}%", 
                f"GAP mediano: {valid_gaps.median():.4f}%", 
                "" 
            ]) 
        if 'optimal_solutions' in df.columns: 
            total_optimal = df['optimal_solutions'].sum() 
            total_runs = df['num_runs'].sum() if 'num_runs' in df.columns else len(df) 
            report_lines.extend([ 
                "=== ANÁLISIS DE OPTIMALIDAD ===", 
                f"Soluciones óptimas encontradas: {total_optimal}", 
                f"Total de ejecuciones: {total_runs}", 
                f"Tasa de éxito: {(total_optimal/total_runs*100):.2f}%", 
                "" 
            ]) 
    elif mode == 'hyperparameters': 
        print(df)
        print(df.columns)
        # Estadísticas de hiperparámetros 
        if 'optimal_solutions' in df.columns: 
            # Filtrar configuraciones con best_gap > 0
            non_zero_opt = df[df['optimal_solutions'] > 0]

            if len(non_zero_opt) > 0:
                # Encontrar el mínimo gap no cero
                min_gap = non_zero_opt['optimal_solutions'].min()
                
                # Obtener todas las configuraciones con ese gap mínimo
                min_gap_configs = non_zero_opt[non_zero_opt['optimal_solutions'] == min_gap]
                
                # Si hay más de una configuración con el mismo gap mínimo, desempatar por mean_gap
                if len(min_gap_configs) > 1:
                    if 'best_gap' in df.columns:
                        # Filtrar configuraciones con best_gap > 0
                        non_zero_gaps = df[df['best_gap'] > 0]

                        if len(non_zero_gaps) > 0:
                            # Encontrar el mínimo gap no cero
                            min_gap = non_zero_gaps['best_gap'].min()
                            
                            # Obtener todas las configuraciones con ese gap mínimo
                            min_gap_configs = non_zero_gaps[non_zero_gaps['best_gap'] == min_gap]
                            
                            # Si hay más de una configuración con el mismo gap mínimo, desempatar por mean_gap
                            if len(min_gap_configs) > 1:
                                # Encontrar la configuración con el menor mean_gap entre los empatados
                                best_config = min_gap_configs.loc[min_gap_configs['std_cost'].idxmin()]
                            else:
                                best_config = min_gap_configs.iloc[0]
                else:
                    best_config = min_gap_configs.iloc[0]
            else:
                # Si todos los gaps son 0, tomar la configuración con mejor mean_gap
                best_config = df.loc[df['best_gap'].idxmin()]
            report_lines.extend([ 
                "=== MEJOR CONFIGURACIÓN ENCONTRADA ===", 
                f"Gap: {best_config['best_gap']:.4f}"
            ]) 
            
            # Añadir parámetros específicos 
            param_cols = [col for col in df.columns if col not in ['instance', 'instance_name', 'algorithm', 'best_cost', 'mean_cost', 'std_cost']] 
            for param in param_cols[:10]: 
                # Limitar a 10 parámetros principales 
                if param in best_config: 
                    report_lines.append(f"{param}: {best_config[param]}") 
            report_lines.append("") 
        # Top 5 configuraciones 
        if len(df) >= 5:
            # Filtrar configuraciones con optimal_solutions > 0
            non_zero_optimal = df[df['optimal_solutions'] > 0]
            
            if len(non_zero_optimal) >= 5:
                # Si hay al menos 5 con optimal_solutions > 0, tomar las top 5
                top5 = non_zero_optimal.nsmallest(5, 'optimal_solutions')
        elif len(non_zero_optimal) > 0:
            # Si hay algunas con optimal_solutions > 0 pero menos de 5, completar con las mejores por best_gap
            remaining_slots = 5 - len(non_zero_optimal)
            top_non_zero = non_zero_optimal.nsmallest(len(non_zero_optimal), 'optimal_solutions')
            
            # Obtener las mejores por best_gap de las que no tienen optimal_solutions > 0
            zero_optimal = df[df['optimal_solutions'] == 0]
            if len(zero_optimal) > 0:
                top_by_gap = zero_optimal.nsmallest(remaining_slots, 'best_gap')
                top5 = pd.concat([top_non_zero, top_by_gap])
            else:
                top5 = top_non_zero
        else:
            # Si no hay ninguna con optimal_solutions > 0, usar best_gap
            top5 = df.nsmallest(5, 'best_gap')
        
            report_lines.extend([
                "=== TOP 5 CONFIGURACIONES ===",
                *[f"{i+1}. Óptimas: {row['optimal_solutions']} - Costo: {row['best_cost']:.4f} - GAP: {row.get('best_gap', 'N/A'):.4f}% - Instancia: {row.get('instance_name', row.get('instance', 'N/A'))}" for i, (_, row) in enumerate(top5.iterrows())],
                ""
            ])
        # Guardar reporte 
        report_path = os.path.join(result_dir, "REPORTE_FINAL.txt") 
        with open(report_path, 'w', encoding='utf-8') as f: 
            f.write('\n'.join(report_lines)) 
            print(f"\nReporte final guardado en: {report_path}") 
            print("\n" + "\n".join(report_lines[:15]) + "...") 
            # Mostrar primeras líneas