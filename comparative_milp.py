from datetime import datetime
import multiprocessing
import os
import json
import pandas as pd
import numpy as np
from algorithms.MILP import milp_ceflp
from main import setup_logger
from structure.create_instances import create_params

manager = multiprocessing.Manager()
lock = manager.Lock()

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

def calculate_gap(primal_cost, optimal_cost):
    """
    Calcula el GAP de optimalidad según la fórmula:
    GAP(%) = 100 * (z_primal - z_dual) / z_primal
    """
    if optimal_cost is None or primal_cost <= 0:
        return None
    
    gap = (primal_cost - optimal_cost) / primal_cost
    return max(0, gap)  # GAP no puede ser negativo

def analyze_results(costs, optimal_cost=None):
    """Analiza los resultados de múltiples ejecuciones"""
    costs_array = np.array(costs)
    
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
        valid_gaps = [g for g in gaps if g is not None]
        
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

def save_compact_results(results_data, result_dir, algorithm_name, search_type="comparative"):
    """Guarda los resultados de forma compacta y clara"""
    
    # Crear directorio principal
    os.makedirs(result_dir, exist_ok=True)
    
    if search_type == "comparative":
        # Para comparativos, guardar resumen por instancia y algoritmo
        summary_file = os.path.join(result_dir, f"{algorithm_name}_comparative_summary.csv")
        
        df = pd.DataFrame(results_data)
        df = df.round(4)  # Redondear a 4 decimales
        df.to_csv(summary_file, index=False)
        
        # Guardar también un archivo detallado con estadísticas
        detailed_file = os.path.join(result_dir, f"{algorithm_name}_comparative_detailed.json")
        with open(detailed_file, 'w') as f:
            json.dump(results_data, f, indent=2)
    
    print(f"Resultados guardados en {result_dir}")

def extract_milp_results(result_dir, algorithm_name, n_nodos):
    """
    Extrae resultados de archivos JSON generados por MILP
    """
    results = []
    json_path = os.path.join(result_dir, f"{n_nodos}.json")

    if os.path.exists(json_path):
        print("WUOA")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(data)
        for instance_key, objective_value in data.items():
            print(objective_value)
            if objective_value != "Infeasible":
                try:
                    cost = float(objective_value)
                    results.append({
                        'algorithm': algorithm_name,
                        'instance_name': instance_key,
                        'n_nodos': n_nodos,
                        'cost': cost,
                        'status': 'Optimal'
                    })
                except ValueError:
                    results.append({
                        'algorithm': algorithm_name,
                        'instance_name': instance_key,
                        'n_nodos': n_nodos,
                        'cost': None,
                        'status': 'Infeasible'
                    })
            else:
                results.append({
                    'algorithm': algorithm_name,
                    'instance_name': instance_key,
                    'n_nodos': n_nodos,
                    'cost': None,
                    'status': 'Infeasible'
                })
    
    return results

def milp_parallel(task_args):
    """
    Función wrapper para ejecutar milp_ceflp con argumentos desempaquetados.
    """
    path, instance, i, n_nodos, result_dir, problem_type, solver, time_limit, log_file_path = task_args

    # Crea el logger dentro del proceso hijo
    log_filename = os.path.join(log_file_path, f"{instance}_execution.log")
    logger = setup_logger(log_filename)
    
    print(f"Procesando instancia {i} de {problem_type} desde {path}...")
    
    try:
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
        print(f"Instancia {i} de {problem_type} finalizada exitosamente.")
        return True
    except Exception as e:
        print(f"Error procesando instancia {i}: {str(e)}")
        return False

def process_comparative_results(base_result_dir, algorithms, optimal_solutions):
    """
    Procesa los resultados para generar métricas comparativas
    """
    all_results = []
    
    for algorithm in algorithms:
        algorithm_dir = os.path.join(base_result_dir, algorithm, "comparative")
        try:
            # Obtener carpetas y ordenar por timestamp
            carpetas = [d for d in os.listdir(algorithm_dir) 
                    if os.path.isdir(os.path.join(algorithm_dir, d))]
            
            # Ordenar por timestamp (asumiendo formato YYYY-MM-DD_HH-MM-SS)
            carpetas.sort(key=lambda x: datetime.strptime(x, "%Y%m%d_%H%M%S"), reverse=True)
            
            # Obtener la más reciente
            carpeta_reciente = carpetas[0] if carpetas else None
            
            if carpeta_reciente:
                directorio_reciente = os.path.join(algorithm_dir, carpeta_reciente)
                print(f"Directorio más reciente: {directorio_reciente}")
                
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
        
        # Extraer resultados para cada tamaño de problema
        for n_nodos in ["n_10", "n_50", "n_100"]:
            results = extract_milp_results(directorio_reciente, algorithm, n_nodos)
            all_results.extend(results)
    
    # Convertir a DataFrame
    df_all = pd.DataFrame(all_results)
    
    if df_all.empty:
        print("No se encontraron resultados para procesar.")
        return None, None
    
    # Filtrar solo resultados válidos (con costo)
    # df_valid = df_all[df_all['cost'].notna()].copy()
    
    # Calcular métricas por algoritmo y tamaño
    summary_data = []
    
    for (algorithm, n_nodos), group in df_all.groupby(['algorithm', 'n_nodos']):
        # Contar total de instancias y casos infeasibles
        total_instances = len(group)
        infeasible_count = len(group[group['status'] == 'Infeasible'])
        feasible_count = total_instances - infeasible_count
        
        # Calcular estadísticas básicas
        # Inicializar estadísticas
        stats = {
            'algorithm': algorithm,
            'n_nodos': n_nodos,
            'total_instances': total_instances,
            'feasible_count': feasible_count,
            'infeasible_count': infeasible_count,
            'feasible_percentage': (feasible_count / total_instances * 100) if total_instances > 0 else 0
        }
        
        # Solo calcular métricas de costos si hay instancias factibles
        if feasible_count > 0:
            df_valid = group[group['cost'].notna()].copy()
            costs = df_valid['cost'].tolist()
            
            stats.update({
                'mean_cost': float(np.mean(costs)),
                'std_cost': float(np.std(costs)),
                'best_cost': float(np.min(costs)),
                'worst_cost': float(np.max(costs)),
            })
            
            # Calcular gaps si hay soluciones óptimas disponibles
            if optimal_solutions:
                gaps = []
                optimal_count = 0
                
                for _, row in df_valid.iterrows():
                    instance_name = row['instance_name']
                    cost = row['cost']
                    optimal_cost = optimal_solutions.get(instance_name)
                    
                    if optimal_cost is not None:
                        try:
                            optimal_cost = float(optimal_cost)
                            gap = calculate_gap(cost, optimal_cost)
                            if gap is not None:
                                gaps.append(gap)
                            
                            # Contar soluciones óptimas (tolerancia 1e-6)
                            if abs(cost - optimal_cost) <= 1e-6:
                                optimal_count += 1
                        except (ValueError, TypeError):
                            continue
                
                if gaps:
                    stats.update({
                        'mean_gap': float(np.mean(gaps)),
                        'std_gap': float(np.std(gaps)),
                        'best_gap': float(np.min(gaps)),
                        'worst_gap': float(np.max(gaps)),
                        'optimal_solutions': optimal_count,
                        'non_optimal_solutions': len(gaps) - optimal_count
                    })
        else:
            # Todas las instancias son infeasibles
            stats.update({
                'mean_cost': None,
                'std_cost': None,
                'best_cost': None,
                'worst_cost': None,
                'mean_gap': None,
                'std_gap': None,
                'best_gap': None,
                'worst_gap': None,
                'optimal_solutions': 0,
                'non_optimal_solutions': 0
            })
        
        summary_data.append(stats)
    
    return summary_data, df_all

def run_experiments():
    global_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Directorio base donde están las instancias
    base_dir = "data/testing_set"
    base_result_dir = "output/"
    os.makedirs(base_result_dir, exist_ok=True)

    # Cargar soluciones óptimas
    print("Cargando soluciones óptimas...")
    optimal_solutions = load_optimal_solutions()
    print(f"Cargadas {len(optimal_solutions)} soluciones óptimas.")
    
    # Configuración del experimento
    nodos = ["n_10", "n_50", "n_100"]
    algoritmos = ["CPLEX"]
    
    # Preparar tareas
    tasks = []
    idx = 0
    
    for algoritmo in algoritmos:
        result_dir = os.path.join(base_result_dir, f"{algoritmo}/comparative/{global_timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        if not os.path.exists(result_dir):
            print(f"[WARNING] Carpeta {result_dir} no encontrada, se ignora.")
            continue
        for nodo in nodos:
            instance_dir = os.path.join(base_dir, nodo)
            if not os.path.exists(instance_dir):
                print(f"[WARNING] Directorio {instance_dir} no encontrado, se omite.")
                continue
                
            for fname in sorted(os.listdir(instance_dir)):
                if fname.endswith(".txt"):
                    path = os.path.join(instance_dir, fname)
                    instance = os.path.splitext(fname)[0]
                    tasks.append((
                        path,                  # ruta al archivo
                        instance,              # nombre de instancia (p{j}_i{x})
                        idx,                   # índice único
                        nodo,                  # identificador del problema (ej: n_10)
                        result_dir,       # carpeta de salida
                        "P2",                  # problema
                        algoritmo,             # optimizador
                        1000,                  # límite de tiempo (1 hora)
                        result_dir
                    ))
                    idx += 1
    
    # Ejecutar experimentos
    if tasks:
        num_processes = min(8, multiprocessing.cpu_count()) 
        print(f"Iniciando la ejecución paralela en {num_processes} procesos...")
        print(f"Total de tareas a ejecutar: {len(tasks)}")
        
        chunk_size = 20  # número de tareas a ejecutar por lote
        total_success = 0
        
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i+chunk_size]
            print(f"Procesando lote {i//chunk_size + 1}/{(len(tasks)-1)//chunk_size + 1}")
            
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(milp_parallel, chunk)
                total_success += sum(1 for r in results if r)
        
        print(f"Experimentos completados: {total_success}/{len(tasks)} exitosos")
    else:
        print("No se encontraron tareas para ejecutar.")
        return
    
    # Procesar y analizar resultados
    print("Procesando resultados y generando métricas...")
    
    try:
        summary_data, df_all = process_comparative_results(base_result_dir, algoritmos, optimal_solutions)
        
        if summary_data:
            # Guardar resultados resumidos
            results_dir = os.path.join(base_result_dir, "analysis")
            save_compact_results(summary_data, results_dir, "MILP_comparative", "comparative")
            
            # Guardar DataFrame completo
            df_all.to_csv(os.path.join(results_dir, "all_milp_results.csv"), index=False)
            
            # Mostrar resumen en consola
            print("\n=== RESUMEN DE RESULTADOS ===")
            for stats in summary_data:
                print(f"\nAlgoritmo: {stats['algorithm']} | Tamaño: {stats['n_nodos']}")
                print(f"  Instancias procesadas: {stats['total_instances']}")
                print(f"  Costo promedio: {stats['mean_cost']:.4f}")
                print(f"  Mejor costo: {stats['best_cost']:.4f}")
                print(f"  Desviación estándar: {stats['std_cost']:.4f}")
                
                if 'mean_gap' in stats:
                    print(f"  GAP promedio: {stats['mean_gap']:.4f}")
                    print(f"  Soluciones óptimas: {stats['optimal_solutions']}")
            
            print(f"\nResultados completos guardados en: {results_dir}")
        
    except Exception as e:
        print(f"Error procesando resultados: {str(e)}")
        print("Revisa los logs para más detalles.")
    
    print(f"Todas las operaciones completadas. Revisa las carpetas en '{base_result_dir}' para los logs y resultados.")

if __name__ == "__main__":
    # run_experiments()
    optimal_solutions = load_optimal_solutions()
    algorithms = ["CPLEX", "GLPK", "CBC"]
    base_result_dir = "output/"
    summary_data, df_all = process_comparative_results(base_result_dir=base_result_dir, algorithms=algorithms, optimal_solutions=optimal_solutions)
    if summary_data:
        # Guardar resultados resumidos
        results_dir = os.path.join(base_result_dir, "analysis")
        save_compact_results(summary_data, results_dir, "MILP_comparative", "comparative")
        
        # Guardar DataFrame completo
        df_all.to_csv(os.path.join(results_dir, "all_milp_results.csv"), index=False)
        
        # Mostrar resumen en consola
        print("\n=== RESUMEN DE RESULTADOS ===")
        for stats in summary_data:
            print(f"\nAlgoritmo: {stats['algorithm']} | Tamaño: {stats['n_nodos']}")
            print(f"  Instancias procesadas: {stats['optimal_solutions']}")
            if stats['mean_cost']:
                print(f"  Costo promedio: {stats['mean_cost']:.4f}")
            if stats['best_cost']:
                print(f"  Mejor costo: {stats['best_cost']:.4f}")
            if stats['std_cost']:
                print(f"  Desviación estándar: {stats['std_cost']:.4f}")
            
            if 'mean_gap' in stats:
                if stats['mean_gap']:
                    print(f"  GAP promedio: {stats['mean_gap']:.4f}")
                if stats['optimal_solutions']:
                    print(f"  Soluciones óptimas: {stats['optimal_solutions']}")
        
        print(f"\nResultados completos guardados en: {results_dir}")