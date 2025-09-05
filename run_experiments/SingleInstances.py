from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import cpu_count
import os
import time

import numpy as np
import pandas as pd

from Orchestrator import calculate_gap, load_optimal_solutions, run_single_algorithm_execution
from structure.create_instances import create_params


def run_fixed_parameters_multiple_instances(task_dict, num_runs=10, num_processes=None):
    """
    Ejecuta un algoritmo con parámetros fijos para UNA instancia y genera resumen.
    """
    algorithm_name = task_dict['algorithm_type']
    fixed_params = {
        k: v for k, v in task_dict.items()
        if k not in ['algorithm_type','params','instance','instance_name','n_nodes','run_idx','temp_dir','problem']
    }

    instance_idx = task_dict['instance']
    instance_name = task_dict['instance_name']
    params = task_dict['params'] if 'params' in task_dict else None
    n_nodes = task_dict['n_nodes']
    problem = task_dict.get('problem')
    result_base_dir = task_dict.get('result_base_dir', './results')

    if num_processes is None:
        num_processes = max(1, min(cpu_count(), num_runs // 2))

    optimal_solutions = load_optimal_solutions()

    print(f"\n=== Ejecutando {algorithm_name.upper()} con parámetros fijos ===")
    print(f"Parámetros: {fixed_params}")
    print(f"Instancia: {instance_name}, Ejecuciones: {num_runs}")
    print(f"Procesos paralelos: {num_processes}")

    start_time = time.time()

    # preparar ejecuciones
    all_results = []
    for run_idx in range(num_runs):
        args = {
            'algorithm_type': algorithm_name,
            'params': params,
            'instance': instance_idx,
            'instance_name': instance_name,
            'n_nodes': n_nodes,
            'run_idx': run_idx,
            'temp_dir': os.path.join('/tmp', f'{algorithm_name}_fixed'),
            'problem': problem,
            **fixed_params
        }
        all_results.append(run_single_algorithm_execution(args))

    # procesar resultados
    processed_results = []
    for result in all_results:
        optimal_cost = optimal_solutions.get(result['instance_name'])
        optimal_cost = round(float(optimal_cost), 2)
        if optimal_cost is not None:
            result['optimal_cost'] = optimal_cost
            result['gap'] = calculate_gap(result['cost'], optimal_cost)
            result['is_optimal'] = int(abs(result['cost'] - optimal_cost) <= 1e-6)
        else:
            result['optimal_cost'] = None
            result['gap'] = None
            result['is_optimal'] = 0
        processed_results.append(result)

    total_time = time.time() - start_time

    # guardar resultados
    result_dir = os.path.join(result_base_dir, f"{algorithm_name}_fixed_params_results")
    os.makedirs(result_dir, exist_ok=True)

    df_results = pd.DataFrame(processed_results)
    results_file = os.path.join(result_dir, f"{algorithm_name}_{instance_name}_results.csv")
    df_results.to_csv(results_file, index=False)

    global_summary = generate_global_summary(processed_results, algorithm_name, n_nodes, fixed_params, total_time)
    save_global_summary(global_summary, result_dir, f"{algorithm_name}_{instance_name}")

    print(f"=== {algorithm_name.upper()} completado ===")
    print(f"Tiempo total: {total_time:.2f} s")
    print(f"Resultados guardados en: {results_file}")


def generate_global_summary(results, algorithm_name, n_nodes, fixed_params, total_time):
    """
    Genera un resumen global de todos los resultados
    """
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        return {
            'error': 'No hay resultados exitosos',
            'total_executions': len(results),
            'successful_executions': 0
        }
    
    # Extraer datos
    all_costs = [r['cost'] for r in successful_results]
    all_times = [r['time'] for r in successful_results]
    all_gaps = [r['gap'] for r in successful_results if r['gap'] is not None]
    optimal_solutions = sum(r['is_optimal'] for r in successful_results)
    
    # Calcular estadísticas globales
    summary = {
        'algorithm': algorithm_name,
        'n_nodes': n_nodes,
        'parameters': fixed_params,
        'execution_info': {
            'total_executions': len(results),
            'successful_executions': len(successful_results),
            'failed_executions': len(results) - len(successful_results),
            'success_rate': len(successful_results) / len(results) * 100,
            'total_time_seconds': total_time,
            'mean_time_per_execution': total_time / len(results),
            'sum_execution_times': sum(all_times)
        },
        'cost_statistics': {
            'solutions_found': len(all_costs),
            'best_cost': float(np.min(all_costs)),
            'worst_cost': float(np.max(all_costs)),
            'mean_cost': float(np.mean(all_costs)),
            'std_cost': float(np.std(all_costs)),
            'median_cost': float(np.median(all_costs))
        },
        'time_statistics': {
            'mean_time': float(np.mean(all_times)),
            'min_time': float(np.min(all_times)),
            'max_time': float(np.max(all_times)),
            'std_time': float(np.std(all_times)),
            'total_computation_time': float(np.sum(all_times))
        },
        'optimality_analysis': {
            'optimal_solutions_found': optimal_solutions,
            'non_optimal_solutions': len(successful_results) - optimal_solutions,
            'optimality_rate': optimal_solutions / len(successful_results) * 100 if successful_results else 0
        }
    }
    
    # Añadir estadísticas de GAP si están disponibles
    if all_gaps:
        summary['gap_statistics'] = {
            'mean_gap': float(np.mean(all_gaps)) * 100,  # Convertir a porcentaje
            'best_gap': float(np.min(all_gaps)) * 100,
            'worst_gap': float(np.max(all_gaps)) * 100,
            'std_gap': float(np.std(all_gaps)) * 100,
            'median_gap': float(np.median(all_gaps)) * 100
        }
    
    # Estadísticas por instancia
    instances_stats = {}
    for result in successful_results:
        instance_name = result['instance_name']
        if instance_name not in instances_stats:
            instances_stats[instance_name] = []
        instances_stats[instance_name].append(result)
    
    summary['instances_count'] = len(instances_stats)
    summary['runs_per_instance'] = len(successful_results) // len(instances_stats) if instances_stats else 0
    
    return summary


def generate_instance_summary(results, optimal_solutions):
    """
    Genera resumen agrupado por instancia
    """
    instance_summaries = []
    
    # Agrupar por instancia
    instances_data = {}
    for result in results:
        if not result['success']:
            continue
        
        instance_name = result['instance_name']
        if instance_name not in instances_data:
            instances_data[instance_name] = []
        instances_data[instance_name].append(result)
    
    # Procesar cada instancia
    for instance_name, instance_results in instances_data.items():
        costs = [r['cost'] for r in instance_results]
        times = [r['time'] for r in instance_results]
        gaps = [r['gap'] for r in instance_results if r['gap'] is not None]
        optimal_count = sum(r['is_optimal'] for r in instance_results)
        optimal_cost = optimal_solutions.get(instance_name)
        
        summary = {
            'instance_name': instance_name,
            'n_nodes': instance_results[0]['n_nodes'],
            'runs': len(instance_results),
            'optimal_cost': optimal_cost,
            
            # Estadísticas de costo
            'best_cost': float(np.min(costs)),
            'worst_cost': float(np.max(costs)),
            'mean_cost': float(np.mean(costs)),
            'std_cost': float(np.std(costs)),
            
            # Estadísticas de tiempo
            'mean_time': float(np.mean(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'sum_time': float(np.sum(times)),
            
            # Optimalidad
            'optimal_solutions': optimal_count,
            'non_optimal_solutions': len(instance_results) - optimal_count,
            'optimality_rate': optimal_count / len(instance_results) * 100
        }
        
        # GAP si está disponible
        if gaps:
            summary.update({
                'mean_gap': float(np.mean(gaps)) * 100,
                'best_gap': float(np.min(gaps)) * 100,
                'worst_gap': float(np.max(gaps)) * 100
            })
        
        instance_summaries.append(summary)
    
    return instance_summaries


def save_global_summary(summary, result_dir, algorithm_name):
    """
    Guarda el resumen global en diferentes formatos
    """
    # Guardar como JSON para máxima información
    json_file = os.path.join(result_dir, f"{algorithm_name}_global_summary.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Guardar reporte de texto legible
    txt_file = os.path.join(result_dir, f"{algorithm_name}_global_report.txt")
    generate_readable_report(summary, txt_file)
    
    print(f"Resumen global guardado en:")
    print(f"  - JSON: {json_file}")
    print(f"  - Reporte: {txt_file}")


def generate_readable_report(summary, output_file):
    """
    Genera un reporte legible en texto plano
    """
    lines = [
        f"=== REPORTE GLOBAL - {summary['algorithm'].upper()} ===",
        f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Número de nodos: {summary['n_nodes']}",
        "",
        "=== PARÁMETROS UTILIZADOS ===",
        *[f"{k}: {v}" for k, v in summary['parameters'].items()],
        "",
        "=== INFORMACIÓN DE EJECUCIÓN ===",
        f"Total de ejecuciones: {summary['execution_info']['total_executions']}",
        f"Ejecuciones exitosas: {summary['execution_info']['successful_executions']}",
        f"Ejecuciones fallidas: {summary['execution_info']['failed_executions']}",
        f"Tasa de éxito: {summary['execution_info']['success_rate']:.2f}%",
        f"Tiempo total: {summary['execution_info']['total_time_seconds']:.2f} segundos",
        f"Tiempo promedio por ejecución: {summary['execution_info']['mean_time_per_execution']:.2f} segundos",
        f"Suma de tiempos de ejecución: {summary['execution_info']['sum_execution_times']:.2f} segundos",
        "",
        "=== ESTADÍSTICAS DE COSTO ===",
        f"Soluciones encontradas: {summary['cost_statistics']['solutions_found']}",
        f"Mejor costo: {summary['cost_statistics']['best_cost']:.4f}",
        f"Peor costo: {summary['cost_statistics']['worst_cost']:.4f}",
        f"Costo promedio: {summary['cost_statistics']['mean_cost']:.4f}",
        f"Desviación estándar: {summary['cost_statistics']['std_cost']:.4f}",
        f"Costo mediano: {summary['cost_statistics']['median_cost']:.4f}",
        "",
        "=== ESTADÍSTICAS DE TIEMPO ===",
        f"Tiempo promedio: {summary['time_statistics']['mean_time']:.4f} segundos",
        f"Tiempo mínimo: {summary['time_statistics']['min_time']:.4f} segundos",
        f"Tiempo máximo: {summary['time_statistics']['max_time']:.4f} segundos",
        f"Desviación estándar tiempo: {summary['time_statistics']['std_time']:.4f} segundos",
        f"Tiempo total de computación: {summary['time_statistics']['total_computation_time']:.4f} segundos",
        "",
        "=== ANÁLISIS DE OPTIMALIDAD ===",
        f"Soluciones óptimas encontradas: {summary['optimality_analysis']['optimal_solutions_found']}",
        f"Soluciones no óptimas: {summary['optimality_analysis']['non_optimal_solutions']}",
        f"Tasa de optimalidad: {summary['optimality_analysis']['optimality_rate']:.2f}%",
        "",
        "=== INFORMACIÓN GENERAL ===",
        f"Número de instancias: {summary['instances_count']}",
        f"Ejecuciones por instancia: {summary['runs_per_instance']}"
    ]
    
    # Añadir estadísticas de GAP si están disponibles
    if 'gap_statistics' in summary:
        lines.extend([
            "",
            "=== ESTADÍSTICAS DE GAP ===",
            f"GAP promedio: {summary['gap_statistics']['mean_gap']:.4f}%",
            f"Mejor GAP: {summary['gap_statistics']['best_gap']:.4f}%",
            f"Peor GAP: {summary['gap_statistics']['worst_gap']:.4f}%",
            f"Desviación estándar GAP: {summary['gap_statistics']['std_gap']:.4f}%",
            f"GAP mediano: {summary['gap_statistics']['median_gap']:.4f}%"
        ])
    
    # Guardar archivo
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    # Mostrar resumen en consola
    print("\n" + "="*50)
    print(f"RESUMEN GLOBAL - {summary['algorithm'].upper()}")
    print("="*50)
    print(f"Instancias: {summary['instances_count']}")
    print(f"Ejecuciones totales: {summary['execution_info']['total_executions']}")
    print(f"Mejor costo: {summary['cost_statistics']['best_cost']:.4f}")
    print(f"Costo promedio: {summary['cost_statistics']['mean_cost']:.4f}")
    if 'gap_statistics' in summary:
        print(f"Mejor GAP: {summary['gap_statistics']['best_gap']:.4f}%")
    print(f"Soluciones óptimas: {summary['optimality_analysis']['optimal_solutions_found']}")
    print(f"Tiempo total: {summary['execution_info']['total_time_seconds']:.2f}s")
    print("="*50)


# Funciones de ejemplo para usar la nueva funcionalidad
def run_grasp_fixed_params(instances_data, n_nodes, result_base_dir, num_runs=10):
    """Ejemplo de uso con GRASP"""
    fixed_params = {
        'alpha': 0.1,
        'frac_neighbors': 6,
        'max_iter': 20
    }
    
    return run_fixed_parameters_multiple_instances(
        algorithm_name='GRASP',
        fixed_params=fixed_params,
        instances_data=instances_data,
        n_nodes=n_nodes,
        result_base_dir=result_base_dir,
        num_runs=num_runs
    )


def run_genetic_fixed_params(instances_data, n_nodes, result_base_dir, num_runs=10):
    """Ejemplo de uso con Genetic Algorithm"""
    fixed_params = {
        'mutation_rate': 0.1,
        'tournament': 4,
        'inicializacion': 'greedy'
    }
    
    return run_fixed_parameters_multiple_instances(
        algorithm_name='GENETIC',
        fixed_params=fixed_params,
        instances_data=instances_data,
        n_nodes=n_nodes,
        result_base_dir=result_base_dir,
        num_runs=num_runs
    )


def run_tabu_fixed_params(instances_data, n_nodes, result_base_dir, num_runs=10):
    """Ejemplo de uso con Tabu Search"""
    fixed_params = {
        'tabu_tenure': 0.25,
        'inicializacion': 'greedy',
        'time_limit': 60
    }
    
    return run_fixed_parameters_multiple_instances(
        algorithm_name='TABU',
        fixed_params=fixed_params,
        instances_data=instances_data,
        n_nodes=n_nodes,
        result_base_dir=result_base_dir,
        num_runs=num_runs
    )