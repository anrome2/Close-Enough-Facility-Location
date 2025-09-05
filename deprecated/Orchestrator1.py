import hashlib
import math
import tempfile
from algorithms.GRASP import GRASPSearch
from algorithms.GeneticAlgorithm import GeneticSearch
from algorithms.TABU import TabuSearch
from structure.create_instances import create_params
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
import numpy as np
import pandas as pd
import time
import os
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import random

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

def calculate_gap(primal_cost, optimal_cost):
    """Calcula el GAP de optimalidad"""
    if optimal_cost is None or primal_cost <= 0:
        return None
    gap = abs((primal_cost - optimal_cost) / primal_cost)
    return max(0, gap)

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
        gaps = [calculate_gap(cost, optimal_cost) for cost in costs]
        valid_gaps = [g for g in gaps if g is not None]
        
        if valid_gaps:
            analysis['mean_gap'] = float(np.mean(valid_gaps))
            analysis['best_gap'] = float(np.min(valid_gaps))
            analysis['std_gap'] = float(np.std(valid_gaps))
        
        tolerance = 1e-6
        optimal_count = sum(1 for cost in costs if abs(cost - optimal_cost) <= tolerance)
        analysis['optimal_solutions'] = optimal_count
        analysis['non_optimal_solutions'] = len(costs) - optimal_count
    
    return analysis

class AlgorithmWrapper(BaseEstimator, RegressorMixin):
    """Wrapper para hacer los algoritmos compatibles con scikit-learn
       Ahora escribe por combinación un CSV con todas las ejecuciones y expone evaluate()."""

    def __init__(self, algorithm_type='GRASP', instance_data=None, n_runs=5, result_dir=None, **params):
        self.algorithm_type = algorithm_type
        self.instance_data = instance_data or []
        self.n_runs = n_runs
        self.params = params
        self.result_dir = result_dir or "results"
        self.optimal_solutions = load_optimal_solutions()
        self._eval_cache = {}

    def _cache_key(self):
        # Clave que identifica una combinación (sin objetos no-hasheables)
        param_items = tuple(sorted(self.params.items()))
        inst_names = tuple(name for name, _ in (self.instance_data or []))
        return (self.algorithm_type, self.n_runs, param_items, inst_names)

    # --- 🔑 Necesarios para que sklearn vea los parámetros ---
    def get_params(self, deep=True):
        out = {
            "algorithm_type": self.algorithm_type,
            "instance_data": self.instance_data,
            "n_runs": self.n_runs,
            "result_dir": self.result_dir,
        }
        out.update(self.params)
        return out

    def set_params(self, **parameters):
        for param, value in parameters.items():
            if param in ["algorithm_type", "instance_data", "n_runs", "result_dir"]:
                setattr(self, param, value)
            else:
                self.params[param] = value
        return self
    # ---------------------------------------------------------
    def evaluate(self, X=None):
        """Ejecuta todas las corridas para esta combinación y devuelve un dict con métricas.
           Además escribe un CSV detallado por combinación con todas las ejecuciones.
        """
        key = (self.algorithm_type, tuple(sorted(self.params.items())), tuple(name for name, _ in self.instance_data), self.n_runs)
        if key in self._eval_cache:
            return self._eval_cache[key]

        # Preparar salida detallada por fila (una fila por instancia x run)
        rows = []
        run_times = []

        # iterar instancias
        for inst_idx, (instance_name, instance_path) in enumerate(self.instance_data):
            # ejecutar n_runs
            costs = []
            times = []
            successes = []
            for run_idx in range(self.n_runs):
                t0 = time.time()
                try:
                    cost = self._single_run(instance_name, instance_path, run_idx)
                    success = not (cost is None or math.isinf(cost) or np.isinf(cost) or cost == float('inf'))
                except Exception:
                    cost = float('inf')
                    success = False
                dt = time.time() - t0

                costs.append(cost)
                times.append(dt)
                run_times.append(dt)
                successes.append(success)

                optimal_cost = self.optimal_solutions.get(instance_name)
                optimal_cost = round(float(optimal_cost), 2)
                gap = calculate_gap(cost, optimal_cost) if optimal_cost is not None and cost is not None and cost != float('inf') else None
                is_optimal = 0
                if optimal_cost is not None and success:
                    if abs(cost - optimal_cost) <= 1e-6:
                        is_optimal = 1

                # Extraer hiperparámetros que interesan (si existen)
                mutation_rate = self.params.get('mutation_rate')
                tournament = self.params.get('tournament')
                inicializacion = self.params.get('inicializacion')
                tabu_tenure = self.params.get('tabu_tenure')
                alpha = self.params.get('alpha')
                frac_neighbors = self.params.get('frac_neighbors')

                row = {
                    "algorithm_type": self.algorithm_type,
                    "instance": inst_idx,
                    "instance_name": instance_name,
                    "n_nodes": self.params.get('n_nodes', None),
                    "combo_key": _params_hash(self.params),   # clave para luego mapear a combo_idx
                    "run_idx": run_idx,
                    "cost": None if cost == float('inf') else float(cost),
                    "time": float(dt),
                    "mutation_rate": mutation_rate,
                    "tournament": tournament,
                    "inicializacion": inicializacion,
                    "tabu_tenure": tabu_tenure,
                    "alpha": alpha,
                    "frac_neighbors": frac_neighbors,
                    "success": bool(success),
                    "optimal_cost": optimal_cost if optimal_cost is not None else None,
                    "gap": float(gap) if gap is not None else None,
                    "is_optimal": int(is_optimal)
                }
                rows.append(row)

        # Escrbir CSV determinista por combinación (hash de params)
        combo_hash = _params_hash(self.params)
        out_dir = os.path.join(self.result_dir, f"{self.algorithm_type}_sklearn_search")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"detailed_runs_{combo_hash}.csv")

        # Escribimos atómicamente (temporal -> replace)
        tmpf = None
        try:
            tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='', suffix='.csv')
            tmpf = tmp.name
            df_rows = pd.DataFrame(rows)
            # Guardar columnas en un orden legible / estable
            cols_order = [
                "algorithm_type","instance","instance_name","n_nodes","combo_key","run_idx",
                "cost","time","mutation_rate","tournament","inicializacion","tabu_tenure",
                "alpha","frac_neighbors","success","optimal_cost","gap","is_optimal"
            ]
            # Si faltan columnas en df_rows, pandas rellenará con NaN
            df_rows.to_csv(tmpf, index=False, columns=cols_order)
            os.replace(tmpf, out_file)
        finally:
            try:
                if tmpf and os.path.exists(tmpf):
                    os.remove(tmpf)
            except:
                pass

        # Analizar resultados por instancia (reusar tu analyze_results)
        per_instance = {}
        for inst_idx, (instance_name, _) in enumerate(self.instance_data):
            # extraer filas pertenecientes a esta instancia
            inst_rows = [r for r in rows if r['instance'] == inst_idx]
            costs = [r['cost'] if r['cost'] is not None else float('inf') for r in inst_rows]
            optimal_cost = self.optimal_solutions.get(instance_name)
            optimal_cost = round(float(optimal_cost), 2)
            analysis = analyze_results(costs, optimal_cost)
            per_instance[instance_name] = analysis

        # Agregadas globales
        total_optimal_runs = sum(per_instance[name].get('optimal_solutions', 0) for name in per_instance)
        num_instances = len(self.instance_data)
        total_runs = num_instances * self.n_runs
        optimal_rate = (total_optimal_runs / total_runs) if total_runs else 0.0
        avg_best_costs = [per_instance[name]['best_cost'] for name in per_instance if 'best_cost' in per_instance[name]]
        avg_best_gap_list = [per_instance[name].get('best_gap') for name in per_instance if per_instance[name].get('best_gap') is not None]
        avg_best_cost = float(np.mean(avg_best_costs)) if avg_best_costs else None
        avg_best_gap = float(np.mean(avg_best_gap_list)) if avg_best_gap_list else None
        avg_time_per_run = float(np.mean(run_times)) if run_times else 0.0

        result = {
            "combo_key": combo_hash,
            "total_optimal_runs": int(total_optimal_runs),
            "total_runs": int(total_runs),
            "optimal_rate": float(optimal_rate),
            "instances_solved_opt": sum(1 for n in per_instance if per_instance[n].get('optimal_solutions', 0) > 0),
            "num_instances": int(num_instances),
            "avg_best_gap": avg_best_gap,
            "avg_best_cost": avg_best_cost,
            "avg_time_per_run": avg_time_per_run,
            "per_instance": per_instance,
            "detailed_rows_count": len(rows),
            "detailed_file": out_file
        }

        self._eval_cache[key] = result
        return result
    
    def fit(self, X=None, y=None):
        return self

    def predict(self, X=None):
        return np.array([0])

    def score(self, X=None, y=None):
        ev = self.evaluate(X)
        # compatibilidad sklearn: si hay gaps, usamos -avg_best_gap, sino -avg_best_cost
        if ev["avg_best_gap"] is not None:
            return -ev["avg_best_gap"]
        if ev["avg_best_cost"] is not None:
            return -ev["avg_best_cost"] / 1000.0
        return -1e6


    def _single_run(self, instance_name, instance_path, run_idx):
        """Ejecuta una sola ejecución del algoritmo"""
        instance_idx = 0  # simplificado
        params = create_params(instance=instance_idx, path=instance_path)

        temp_dir = os.path.join('/tmp', f'{self.algorithm_type}_sklearn_run_{run_idx}')
        os.makedirs(temp_dir, exist_ok=True)

        logger = logging.getLogger(f'{self.algorithm_type}_sklearn_{run_idx}')
        logger.setLevel(logging.WARNING)

        try:
            if self.algorithm_type == 'GRASP':
                algorithm = GRASPSearch(
                    params=params,
                    instance=instance_name,
                    result_dir=temp_dir,
                    alpha=self.params.get('alpha', 0.3),
                    frac_neighbors=self.params.get('frac_neighbors', 4),
                    logger=logger,
                    problem=self.params.get('problem', 'P2'),
                    max_iter=self.params.get('max_iter', None)
                )

            elif self.algorithm_type == 'GENETIC':
                algorithm = GeneticSearch(
                    params=params,
                    instance=instance_idx,
                    problem=self.params.get('problem', 'P2'),
                    result_dir=temp_dir,
                    inicializacion=self.params.get('inicializacion', 'random'),
                    mutation_rate=self.params.get('mutation_rate', 0.1),
                    tournament=self.params.get('tournament', 5),
                    logger=logger
                )

            elif self.algorithm_type == 'TABU':
                algorithm = TabuSearch(
                    params=params,
                    inicializacion=self.params.get('inicializacion', 'random'),
                    instance=instance_idx,
                    problem=self.params.get('problem', 'P2'),
                    result_dir=temp_dir,
                    tabu_tenure=self.params.get('tabu_tenure', 0.3),
                    time_limit=self.params.get('time_limit', 50),
                    logger=logger
                )

            algorithm.run()

            if hasattr(algorithm, 'best_solution'):
                return round(algorithm.best_solution.cost, 2)
            elif hasattr(algorithm, 'best_individual'):
                return round(algorithm.best_individual.cost, 2)
            else:
                return float('inf')

        except Exception:
            return float('inf')
        finally:
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass

def scorer_opt_runs(est, X, y=None):
    return est.evaluate(X)["total_optimal_runs"]

def scorer_opt_rate(est, X, y=None):
    return est.evaluate(X)["optimal_rate"]

def scorer_neg_best_gap(est, X, y=None):
    g = est.evaluate(X)["avg_best_gap"]
    return (-g) if g is not None else -1e-12

def scorer_neg_best_cost(est, X, y=None):
    val = est.evaluate(X)["avg_best_cost"]
    return -val / 1000.0 if val is not None else -1e12

def _params_hash(params: dict) -> str:
    """Hash determinista para un diccionario de parámetros (clave para archivos)."""
    # serializar ordenado, convertir objetos no serializables a str
    as_list = sorted((k, params[k]) for k in params)
    j = json.dumps(as_list, default=str, sort_keys=True)
    return hashlib.md5(j.encode('utf-8')).hexdigest()

def gap_scorer(estimator, X=None, y=None):
    """Custom scorer para GAP"""
    return estimator.score(X, y)

def run_sklearn_hyperparameter_search(algorithm_name, param_space, instances_data, n_nodes,
                                     result_base_dir, search_type='grid', n_iter=50, 
                                     num_runs=5, cv=2, n_jobs=None):
    """
    Versión extendida que genera:
      - detailed per-run CSVs por combinación (escritos por AlgorithmWrapper.evaluate)
      - all_runs_<ALGO>_detailed.csv
      - global_stats_<ALGO>.csv
      - hyperparameters_summary_<ALGO>.csv
      - stats_by_instance_<ALGO>.csv
      - top10_hyperparameters_<ALGO>.csv
    """
    if n_jobs is None:
        n_jobs = min(cpu_count(), 8)

    print(f"\n=== Búsqueda de hiperparámetros {algorithm_name.upper()} (detailed outputs) ===")
    print(f"Tipo de búsqueda: {search_type}")
    print(f"Instancias: {len(instances_data)}")
    print(f"Ejecuciones por combinación (n_runs): {num_runs}")
    print(f"Procesos paralelos (n_jobs): {n_jobs}")
    start_time = time.time()

    estimator = AlgorithmWrapper(
        algorithm_type=algorithm_name,
        instance_data=instances_data,
        n_runs=num_runs,
        result_dir=result_base_dir
    )

    # métricas múltiples
    scoring = {
        "opt_runs": scorer_opt_runs,
        "opt_rate": scorer_opt_rate,
        "neg_best_gap": scorer_neg_best_gap,
        "neg_best_cost": scorer_neg_best_cost
    }

    if search_type == 'grid':
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_space,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1,
            refit=False
        )
    else:
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_space,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1,
            random_state=42,
            refit=False
        )

    # X con tantas filas como instancias
    X_dummy = np.arange(len(instances_data)).reshape(-1, 1)
    search.fit(X_dummy)

    # -------------------------------------------------
    # 1) recoger resultados sklearn y localizar los CSV detallados escritos por evaluate()
    # -------------------------------------------------
    results_df = pd.DataFrame(search.cv_results_)

    # directorio donde los evaluadores guardaron los detallados
    result_dir = os.path.join(result_base_dir, f"{algorithm_name}_sklearn_search")
    os.makedirs(result_dir, exist_ok=True)

    # Para cada fila de cv_results_ intentamos leer el fichero detailed_runs_{combo_hash}.csv
    detailed_frames = []
    combo_keys = []
    for idx, row in results_df.iterrows():
        params = row['params']
        combo_hash = _params_hash(params)
        combo_keys.append(combo_hash)
        detailed_file = os.path.join(result_dir, f"detailed_runs_{combo_hash}.csv")
        if os.path.exists(detailed_file):
            df = pd.read_csv(detailed_file)
            # anotamos combo_idx (orden en results_df) para facilitar filtrado posterior
            df['combo_idx'] = int(idx)
            # anotar también los parámetros por columna si faltan
            for k, v in params.items():
                colname = f"param_{k}"
                df[colname] = v
            detailed_frames.append(df)
        else:
            # Si falta el fichero, agregamos una fila vacía con params para debug
            print(f"[WARNING] No se encontró detailed file para params hash {combo_hash} (idx {idx}).")

    if detailed_frames:
        all_runs_df = pd.concat(detailed_frames, ignore_index=True, sort=False)
    else:
        all_runs_df = pd.DataFrame(columns=[
            "algorithm_type","instance","instance_name","n_nodes","combo_key","run_idx",
            "cost","time","mutation_rate","tournament","inicializacion","tabu_tenure",
            "alpha","frac_neighbors","success","optimal_cost","gap","is_optimal","combo_idx"
        ])

    # Renombrar columnas para que coincidan con tu formato deseado:
    # 'instance' es índice; 'instance_name' ya está.
    # Añadir columna n_nodes: si no existe en filas, poner el n_nodes recibido por la llamada
    if 'n_nodes' not in all_runs_df.columns or all_runs_df['n_nodes'].isnull().all():
        all_runs_df['n_nodes'] = n_nodes

    # Reordenamos columnas y mantenemos solo las solicitadas explícitamente:
    out_cols = ["algorithm_type","instance","instance_name","n_nodes","combo_idx","run_idx",
                "cost","time","mutation_rate","tournament","inicializacion","success",
                "optimal_cost","gap","is_optimal"]
    for c in out_cols:
        if c not in all_runs_df.columns:
            all_runs_df[c] = None
    all_runs_df = all_runs_df[out_cols]

    # Guardar el CSV 1 (todas las soluciones de las instancias)
    all_runs_file = os.path.join(result_dir, f"all_runs_{algorithm_name}_detailed.csv")
    all_runs_df.to_csv(all_runs_file, index=False)

    # -------------------------------------------------
    # 2) Global statics: agrupar por (mutation_rate, tournament, inicializacion, n_nodes)
    # -------------------------------------------------
    gb_cols = ['mutation_rate','tournament','inicializacion','n_nodes']
    # Asegurarnos de que esas columnas existan
    for c in gb_cols:
        if c not in all_runs_df.columns:
            all_runs_df[c] = None

    # convertir tipos numéricos
    all_runs_df['cost'] = pd.to_numeric(all_runs_df['cost'], errors='coerce')
    all_runs_df['time'] = pd.to_numeric(all_runs_df['time'], errors='coerce')
    all_runs_df['gap'] = pd.to_numeric(all_runs_df['gap'], errors='coerce')
    all_runs_df['is_optimal'] = pd.to_numeric(all_runs_df['is_optimal'], errors='coerce').fillna(0).astype(int)

    g = all_runs_df.groupby(gb_cols)
    global_stats = g['cost'].agg(['mean','std','min','max']).rename(columns={
        'mean':'cost_mean','std':'cost_std','min':'cost_min','max':'cost_max'
    })
    times = g['time'].agg(['mean','sum']).rename(columns={'mean':'time_mean','sum':'time_sum'})
    gaps = g['gap'].agg(['mean','sum']).rename(columns={'mean':'gap_mean','sum':'gap_sum'})
    opt_sums = g['is_optimal'].agg(['sum']).rename(columns={'sum':'is_optimal_sum'})

    global_stats_df = pd.concat([global_stats, times, gaps, opt_sums], axis=1).reset_index()
    global_stats_file = os.path.join(result_dir, f"global_stats_{algorithm_name}.csv")
    global_stats_df.to_csv(global_stats_file, index=False)

    # -------------------------------------------------
    # 3) hyperparameters summary (por combo_idx x instance)
    # -------------------------------------------------
    # Partimos de all_runs_df; agregamos por combo_idx y instance
    hp_groups = all_runs_df.groupby(['combo_idx','instance','instance_name','n_nodes'])
    summary_rows = []
    for name, group in hp_groups:
        combo_idx, inst_idx, inst_name, n_nodes_v = name
        costs = group['cost'].dropna().tolist()
        times = group['time'].dropna().tolist()
        optimal_cost = group['optimal_cost'].dropna().unique()
        optimal_cost_val = float(optimal_cost[0]) if len(optimal_cost) > 0 else None

        mean_cost = float(np.mean(costs)) if costs else float('nan')
        std_cost = float(np.std(costs)) if costs else float('nan')
        best_cost = float(np.min(costs)) if costs else float('nan')
        worst_cost = float(np.max(costs)) if costs else float('nan')
        num_runs = len(costs)
        # contar óptimos
        optimal_solutions = int(group['is_optimal'].sum())
        non_optimal_solutions = int(num_runs - optimal_solutions)
        # gaps (por run)
        gaps = group['gap'].dropna().tolist()
        mean_gap = float(np.mean(gaps)) if gaps else None
        best_gap = float(np.min(gaps)) if gaps else None
        std_gap = float(np.std(gaps)) if gaps else None
        mean_time = float(np.mean(times)) if times else None

        # tomar params del grupo (las columnas param_*)
        params = {}
        for c in group.columns:
            if c.startswith('param_'):
                params[c[len('param_'):]] = group.iloc[0][c]

        row = {
            "combo_idx": int(combo_idx),
            "instance": int(inst_idx),
            "instance_name": inst_name,
            "n_nodes": n_nodes_v,
            "algorithm": algorithm_name,
            "algorithm_type": algorithm_name,
            **params,
            "mean_cost": mean_cost,
            "std_cost": std_cost,
            "best_cost": best_cost,
            "worst_cost": worst_cost,
            "num_runs": num_runs,
            "optimal_solutions": optimal_solutions,
            "non_optimal_solutions": non_optimal_solutions,
            "mean_gap": mean_gap,
            "best_gap": best_gap,
            "std_gap": std_gap,
            "mean_time": mean_time,
            "optimal_value": optimal_cost_val
        }
        summary_rows.append(row)

    hyperparameters_summary_df = pd.DataFrame(summary_rows)
    print(hyperparameters_summary_df)
    hyperparameters_summary_file = os.path.join(result_dir, f"hyperparameters_summary_{algorithm_name}.csv")
    hyperparameters_summary_df.to_csv(hyperparameters_summary_file, index=False)

    # -------------------------------------------------
    # 4) stats by instance: agregación sobre todas las combinaciones por instancia
    # -------------------------------------------------
    inst_g = hyperparameters_summary_df.groupby('instance')
    stats_inst_rows = []
    for inst, gdf in inst_g:
        mean_best = float(gdf['best_cost'].mean()) if not gdf['best_cost'].isnull().all() else None
        min_best = float(gdf['best_cost'].min()) if not gdf['best_cost'].isnull().all() else None
        max_best = float(gdf['best_cost'].max()) if not gdf['best_cost'].isnull().all() else None
        mean_cost = float(gdf['mean_cost'].mean()) if not gdf['mean_cost'].isnull().all() else None
        mean_gap = float(gdf['mean_gap'].dropna().mean()) if not gdf['mean_gap'].dropna().empty else None

        stats_inst_rows.append({
            "instance": int(inst),
            "best_cost_mean": mean_best,
            "best_cost_min": min_best,
            "best_cost_max": max_best,
            "mean_cost_mean": mean_cost,
            "mean_gap_mean": mean_gap
        })

    stats_by_instance_df = pd.DataFrame(stats_inst_rows).set_index('instance')
    stats_by_instance_file = os.path.join(result_dir, f"stats_by_instance_{algorithm_name}.csv")
    stats_by_instance_df.to_csv(stats_by_instance_file)

    # -------------------------------------------------
    # 5) top 10 hyperparameter lines (ordenadas por criterio principal)
    # -------------------------------------------------
    # Ordenamos hyperparameters_summary_df: primero por optimal_solutions desc, luego optimal_rate, luego mean_gap asc, mean_cost asc
    if 'optimal_solutions' in hyperparameters_summary_df.columns:
        top_sorted = hyperparameters_summary_df.sort_values(
            by=['optimal_solutions','mean_gap','mean_cost'],
            ascending=[False, True, True]
        )
    else:
        top_sorted = hyperparameters_summary_df.sort_values(by=['mean_cost'], ascending=True)

    top10_df = top_sorted.head(10)
    top10_file = os.path.join(result_dir, f"top10_hyperparameters_{algorithm_name}.csv")
    top10_df.to_csv(top10_file, index=False)

    # ---------------- summary json ----------------
    # Elegir la mejor fila de results_df según la métrica 'mean_test_opt_runs' si existe
    if 'mean_test_opt_runs' in results_df.columns:
        best_row = results_df.sort_values('mean_test_opt_runs', ascending=False).iloc[0]
        best_params = best_row['params']
        best_opt_runs = float(best_row['mean_test_opt_runs'])
        best_opt_rate = float(best_row.get('mean_test_opt_rate', np.nan))
    else:
        best_params = results_df.iloc[0]['params'] if not results_df.empty else None
        best_opt_runs = None
        best_opt_rate = None

    total_time = time.time() - start_time
    summary = {
        'algorithm': algorithm_name,
        'search_type': search_type,
        'total_combinations': int(len(results_df)),
        'best_params': best_params,
        'best_optimal_runs': best_opt_runs,
        'best_optimal_rate': best_opt_rate,
        'total_time': total_time,
        'avg_time_per_combination': total_time / len(results_df) if len(results_df) else None,
        'all_runs_file': all_runs_file,
        'global_stats_file': global_stats_file,
        'hyperparameters_summary_file': hyperparameters_summary_file,
        'stats_by_instance_file': stats_by_instance_file,
        'top10_file': top10_file
    }

    summary_file = os.path.join(result_dir, f"{algorithm_name}_sklearn_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Búsqueda completada ({algorithm_name}) ===")
    print(f"Resultados guardados en: {result_dir}")
    return global_stats_df, summary

# Espacios de parámetros para cada algoritmo
GRASP_PARAM_SPACE = {
    'alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
    'frac_neighbors': [2, 4, 6, 8],
    'problem': ['P2']
}

GRASP_PARAM_SPACE_RANDOM = {
    'alpha': np.arange(0.05, 0.95, 0.05),
    'frac_neighbors': range(2, 9),
    'problem': ['P2']
}

GENETIC_PARAM_SPACE = {
    'mutation_rate': [0.01, 0.05, 0.1,],
    'tournament': [3, 5, 7, 10],
    'inicializacion': ['random', 'greedy'],
    'problem': ['P2']
}

GENETIC_PARAM_SPACE_RANDOM = {
    'mutation_rate': [0.01, 0.05, 0.1,],
    'tournament': [3, 5, 7, 10],
    'inicializacion': ['random', 'greedy'],
    'problem': ['P2']
}

TABU_PARAM_SPACE = {
    'tabu_tenure': [0.25, 0.5, 0.75],
    'inicializacion': ['random', 'greedy', 'kmeans'],
    'time_limit': [10, 50, 120],
    'problem': ['P2']
}

TABU_PARAM_SPACE_RANDOM = {
    'tabu_tenure': [0.25, 0.5, 0.75],
    'inicializacion': ['random', 'greedy', 'kmeans'],
    'time_limit': [10, 50, 120],
    'problem': ['P2']
}

# Funciones de compatibilidad (mantener el resto de tu código igual)
def run_committee_multiple_instances(algorithm_params_list, instances_data, n_nodes, result_base_dir, 
                                  algorithm_name, num_runs=5, num_processes=None):
    """Mantener esta función igual para los comités"""
    # ... tu implementación actual ...
    pass

