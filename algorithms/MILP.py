import json
import os
import time
import pulp
import numpy as np
import scipy.sparse as sp
from collections import defaultdict


def preprocess_distance_matrices(d_ij, d_kj, sparsity_threshold=0.7):
    """
    Preprocesa las matrices de distancia para optimizar el rendimiento.
    
    Args:
        d_ij: Matriz de distancias cliente-instalación
        d_kj: Matriz de distancias punto_recogida-instalación
        sparsity_threshold: Umbral para convertir a matriz dispersa (porcentaje de ceros)
    
    Returns:
        Tupla con las matrices preprocesadas y información adicional
    """
    preprocessing_info = {}
    
    # Convertir a numpy arrays si no lo son
    if not isinstance(d_ij, (np.ndarray, sp.spmatrix)):
        d_ij_array = np.array(d_ij)
    else:
        d_ij_array = d_ij
        
    if not isinstance(d_kj, (np.ndarray, sp.spmatrix)):
        d_kj_array = np.array(d_kj)
    else:
        d_kj_array = d_kj
    
    # Calcular estadísticas de dispersión
    d_ij_zeros = np.count_nonzero(d_ij_array == 0) / d_ij_array.size
    d_kj_zeros = np.count_nonzero(d_kj_array == 0) / d_kj_array.size
    
    preprocessing_info['d_ij_sparsity'] = d_ij_zeros
    preprocessing_info['d_kj_sparsity'] = d_kj_zeros
    
    # Convertir a matrices dispersas si son muy dispersas
    if d_ij_zeros > sparsity_threshold:
        d_ij_processed = sp.csr_matrix(d_ij_array)
        preprocessing_info['d_ij_sparse'] = True
    else:
        d_ij_processed = d_ij_array
        preprocessing_info['d_ij_sparse'] = False
        
    if d_kj_zeros > sparsity_threshold:
        d_kj_processed = sp.csr_matrix(d_kj_array)
        preprocessing_info['d_kj_sparse'] = True
    else:
        d_kj_processed = d_kj_array
        preprocessing_info['d_kj_sparse'] = False
    
    return d_ij_processed, d_kj_processed, preprocessing_info


def filter_valid_combinations(I, J, K, K_i, I_k, d_ij, d_kj, h, 
                             distance_threshold=None, 
                             demand_threshold=None,
                             min_clients_per_pickup=1):
    """
    Filtra combinaciones válidas basándose en criterios de factibilidad.
    
    Args:
        I, J, K: Conjuntos de clientes, instalaciones y puntos de recogida
        K_i, I_k: Mapeos de clientes a puntos válidos y viceversa
        d_ij, d_kj: Matrices de distancias
        h: Demandas de clientes
        distance_threshold: Distancia máxima permitida
        demand_threshold: Demanda mínima para considerar una combinación
        min_clients_per_pickup: Mínimo número de clientes por punto de recogida
    
    Returns:
        Diccionario con combinaciones válidas filtradas
    """
    valid_combinations = {
        'x_indices': [],  # (i, j) válidas
        'z_indices': [],  # (i, k) válidas
        's_indices': [],  # (k, j) válidas
        'w_indices': [],  # (i, k, j) válidas para P2
    }
    
    # Filtrar combinaciones x (cliente-instalación directa)
    for i in I:
        for j in J:
            if i != j:  # No auto-asignación
                # Obtener distancia
                if isinstance(d_ij, (list, tuple)):
                    dist = d_ij[min(i,j)][max(i,j)]
                else:
                    dist = d_ij[i-1, j-1] if hasattr(d_ij, 'shape') else d_ij[i][j]
                
                # Aplicar filtros
                valid = True
                if distance_threshold is not None and dist > distance_threshold:
                    valid = False
                if demand_threshold is not None and h[i-1] < demand_threshold:
                    valid = False
                
                if valid:
                    valid_combinations['x_indices'].append((i, j))
    
    # Filtrar combinaciones z (cliente-punto de recogida)
    for i in I:
        for k in K_i[i]:
            # Verificar que el punto de recogida tenga suficientes clientes potenciales
            if len(I_k[k]) >= min_clients_per_pickup:
                valid_combinations['z_indices'].append((i, k))
    
    # Filtrar combinaciones s (punto de recogida-instalación)
    for k in K:
        # Solo incluir puntos de recogida con al menos un cliente válido
        if len([i for i in I_k[k] if (i, k) in [(x[0], x[1]) for x in valid_combinations['z_indices']]]) > 0:
            for j in J:
                # Aplicar filtro de distancia si existe
                if distance_threshold is not None:
                    if isinstance(d_kj, (list, tuple)):
                        dist = d_kj[k][j]
                    else:
                        dist = d_kj[k-1, j-1] if hasattr(d_kj, 'shape') else d_kj[k][j]
                    
                    if dist <= distance_threshold:
                        valid_combinations['s_indices'].append((k, j))
                else:
                    valid_combinations['s_indices'].append((k, j))
    
    # Para P2: filtrar combinaciones w
    for i in I:
        for k_or_i in K_i[i] + [i]:
            for j in J:
                # Verificar si la combinación es válida
                if k_or_i == i:
                    # Asignación directa: verificar si (i,j) es válida
                    if (i, j) in valid_combinations['x_indices']:
                        valid_combinations['w_indices'].append((i, k_or_i, j))
                else:
                    # Vía punto de recogida: verificar si (i,k) y (k,j) son válidas
                    if ((i, k_or_i) in valid_combinations['z_indices'] and 
                        (k_or_i, j) in valid_combinations['s_indices']):
                        valid_combinations['w_indices'].append((i, k_or_i, j))
    
    return valid_combinations


def optimize_solver_settings(optimizer="CBC", time_limit=None, gap_rel=0.01):
    """
    Configurar el solver para mejor rendimiento.
    
    Args:
        optimizer: Tipo de solver ("CBC", "GLPK", "CPLEX")
        time_limit: Límite de tiempo en segundos
        gap_rel: Gap de optimalidad relativo
    
    Returns:
        Solver configurado
    """
    if optimizer == "CBC":
        solver = pulp.PULP_CBC_CMD(
            msg=True,
            timeLimit=time_limit,
            gapRel=gap_rel,
            threads=0,  # Usar todos los cores disponibles
            options=[
                'cuts', 'on',
                'heur', 'on',
                'preprocess', 'on',
                'gomory', 'on',
                'knapsack', 'on',
                'probing', 'on',
                'clique', 'on'
            ]
        )
    elif optimizer == "GLPK":
        options = ['--cuts']
        if time_limit:
            options.extend(['--tmlim', str(time_limit)])
        solver = pulp.GLPK_CMD(
            msg=True,
            options=options,
            timeLimit=time_limit
        )
    elif optimizer == "CPLEX":
        solver = pulp.CPLEX_CMD(
            path="/home/andrea/Documentos/Close-Enough-Facility-Location/cplex/bin/x86-64_linux/cplex",
            msg=True,
            timeLimit=time_limit,
            options=['set mip tolerances mipgap ' + str(gap_rel)]
        )

    else:
        # Fallback a CBC
        solver = optimize_solver_settings("CBC", time_limit, gap_rel)
    
    return solver


def create_optimized_model_p1(I, J, K, K_i, I_k, h, d_ij, d_kj, p, t, M_param, valid_combinations):
    """
    Crear modelo P1 optimizado.
    """
    model = pulp.LpProblem("P1_CEFLP", pulp.LpMinimize)
    
    # Variables optimizadas - solo crear las necesarias
    x = pulp.LpVariable.dicts("x", valid_combinations['x_indices'], cat=pulp.LpBinary)
    z = pulp.LpVariable.dicts("z", valid_combinations['z_indices'], cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", J, cat=pulp.LpBinary)
    nu = pulp.LpVariable.dicts("nu", K, cat=pulp.LpBinary)
    s = pulp.LpVariable.dicts("s", valid_combinations['s_indices'], lowBound=0, cat=pulp.LpContinuous)
    
    # Función objetivo optimizada
    obj_terms = []
    
    # Términos de asignación directa
    for (i, j) in valid_combinations['x_indices']:
        if isinstance(d_ij, (list, tuple)):
            dist = d_ij[min(i,j)][max(i,j)]
        else:
            dist = d_ij[i-1, j-1] if hasattr(d_ij, 'shape') else d_ij[i][j]
        
        coeff = h[i-1] * dist
        if coeff != 0:
            obj_terms.append(coeff * x[(i, j)])
    
    # Términos de distribución vía puntos de recogida
    for (k, j) in valid_combinations['s_indices']:
        if isinstance(d_kj, (list, tuple)):
            dist = d_kj[k][j]
        else:
            dist = d_kj[k-1, j-1] if hasattr(d_kj, 'shape') else d_kj[k][j]
        
        if dist != 0:
            obj_terms.append(dist * s[(k, j)])
    
    model += pulp.lpSum(obj_terms)
    
    # Restricciones optimizadas
    # Restricción (3): Se abren exactamente p instalaciones
    model += pulp.lpSum([y[j] for j in J]) == p
    
    # Restricción (4): Se abren exactamente t puntos de recogida
    model += pulp.lpSum([nu[k] for k in K]) == t
    
    # Restricción (5): Cada cliente se asigna a una instalación o punto de recogida
    for i in I:
        x_terms = [x[(i, j)] for (i_x, j) in valid_combinations['x_indices'] if i_x == i]
        z_terms = [z[(i, k)] for (i_z, k) in valid_combinations['z_indices'] if i_z == i]
        model += pulp.lpSum(x_terms + z_terms) == 1
    
    # Restricción (6): Conservación de flujo en puntos de recogida
    for k in K:
        supply_terms = [s[(k, j)] for (k_s, j) in valid_combinations['s_indices'] if k_s == k]
        demand_terms = [h[i-1] * z[(i, k)] for (i_z, k_z) in valid_combinations['z_indices'] 
                       if k_z == k for i in [i_z]]
        
        if supply_terms and demand_terms:
            model += pulp.lpSum(supply_terms) == pulp.lpSum(demand_terms)
    
    # Restricción (7): Envío solo si instalación está abierta
    for j in J:
        supply_to_j = [s[(k, j)] for (k, j_s) in valid_combinations['s_indices'] if j_s == j]
        if supply_to_j:
            model += pulp.lpSum(supply_to_j) <= M_param * y[j]
    
    # Restricción (8): Asignación a punto de recogida solo si está abierto
    for (i, k) in valid_combinations['z_indices']:
        model += z[(i, k)] <= nu[k]
    
    # Restricción (9): Asignación directa solo si instalación está abierta
    for (i, j) in valid_combinations['x_indices']:
        model += x[(i, j)] <= y[j]
    
    return model, {'x': x, 'z': z, 'y': y, 'nu': nu, 's': s}


def create_optimized_model_p2(I, J, K, K_i, I_k, h, d_ij, d_kj, p, t, valid_combinations):
    """
    Crear modelo P2 optimizado.
    """
    model = pulp.LpProblem("P2_CEFLP", pulp.LpMinimize)
    
    # Variables P2 optimizadas
    w = pulp.LpVariable.dicts("w", valid_combinations['w_indices'], cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", J, cat=pulp.LpBinary)
    nu = pulp.LpVariable.dicts("nu", K, cat=pulp.LpBinary)
    
    # Función objetivo P2 optimizada
    obj_terms = []
    for (i, k_or_i, j) in valid_combinations['w_indices']:
        if k_or_i == i and i != j:
            dist = d_ij[min(i,j)][max(i,j)]
        elif k_or_i != i and i != j:
            dist = d_kj[k_or_i][j]
        else:
            dist = 0
        
        coeff = h[i-1] * dist
        if coeff != 0:
            obj_terms.append(coeff * w[(i, k_or_i, j)])
    
    model += pulp.lpSum(obj_terms)
    
    # Restricciones P2
    # Restricción (19): Se abren exactamente p instalaciones
    model += pulp.lpSum([y[j] for j in J]) == p
    
    # Restricción (20): Se abren exactamente t puntos de recogida
    model += pulp.lpSum([nu[k] for k in K]) == t
    
    # Restricción (21): Cada cliente debe ser asignado a exactamente una combinación
    for i in I:
        terms = [w[(i, k_or_i, j)] for (i_w, k_or_i, j) in valid_combinations['w_indices'] if i_w == i]
        model += pulp.lpSum(terms) == 1
    
    # Restricción (22): w_ikj <= y_j
    for (i, k_or_i, j) in valid_combinations['w_indices']:
        model += w[(i, k_or_i, j)] <= y[j]
    
    # Restricción (23): w_ikj <= nu_k (solo para puntos de recogida reales)
    for (i, k_or_i, j) in valid_combinations['w_indices']:
        if k_or_i != i:  # Solo puntos de recogida reales
            model += w[(i, k_or_i, j)] <= nu[k_or_i]
    
    return model, {'w': w, 'y': y, 'nu': nu}


def milp_ceflp(params, instance, n_nodos, result_dir, logger, problem_type="P1", 
               optimizer="CBC", time_limit=None, optimal_solution=False, 
               use_preprocessing=False, distance_threshold=None, gap_rel=0.01):
    """
    Versión optimizada de la función milp_ceflp.
    
    Args adicionales:
        use_preprocessing: Usar preprocesamiento de matrices y filtrado
        distance_threshold: Umbral de distancia para filtrar combinaciones
        gap_rel: Gap de optimalidad relativo para el solver
    """
    # Extraer parámetros
    I = params['I']
    J = params['J']
    K = params['K']
    h = params['h']
    d_ij = params['d_ij']
    d_kj = params['d_kj']
    K_i = params['K_i']
    I_k = params['I_k']
    p = params['p']
    t = params['t']
    
    # Crear carpeta de output si no existe
    os.makedirs(result_dir, exist_ok=True)
    
    if optimal_solution:
        log_dir = os.path.join(result_dir, n_nodos)
    else:
        log_dir = result_dir
    
    # Preprocesamiento
    preprocessing_start = time.time()
    
    if use_preprocessing:
        logger.debug("Iniciando preprocesamiento...")
        
        # Preprocesar matrices de distancia
        d_ij_processed, d_kj_processed, preprocessing_info = preprocess_distance_matrices(d_ij, d_kj)
        logger.debug(f"Dispersión d_ij: {preprocessing_info['d_ij_sparsity']:.2%}")
        logger.debug(f"Dispersión d_kj: {preprocessing_info['d_kj_sparsity']:.2%}")
        
        # Filtrar combinaciones válidas
        valid_combinations = filter_valid_combinations(
            I, J, K, K_i, I_k, d_ij_processed, d_kj_processed, h,
            distance_threshold=distance_threshold
        )
        
        logger.debug(f"Combinaciones válidas:")
        logger.debug(f"  x_indices: {len(valid_combinations['x_indices'])}")
        logger.debug(f"  z_indices: {len(valid_combinations['z_indices'])}")
        logger.debug(f"  s_indices: {len(valid_combinations['s_indices'])}")
        if problem_type == "P2":
            logger.debug(f"  w_indices: {len(valid_combinations['w_indices'])}")
        
    else:
        # Sin preprocesamiento: usar todas las combinaciones
        d_ij_processed, d_kj_processed = d_ij, d_kj
        valid_combinations = {
            'x_indices': [(i, j) for i in I for j in J if i != j],
            'z_indices': [(i, k) for i in I for k in K_i[i]],
            's_indices': [(k, j) for k in K for j in J],
            'w_indices': [(i, k_or_i, j) for i in I for k_or_i in K_i[i] + [i] for j in J]
        }
    
    preprocessing_time = time.time() - preprocessing_start
    logger.debug(f"Tiempo de preprocesamiento: {preprocessing_time:.4f} segundos")
    
    # Crear modelo
    model_start = time.time()
    
    if problem_type == "P1":
        M_param = params['M']
        model, variables = create_optimized_model_p1(
            I, J, K, K_i, I_k, h, d_ij_processed, d_kj_processed, 
            p, t, M_param, valid_combinations
        )
    elif problem_type == "P2":
        model, variables = create_optimized_model_p2(
            I, J, K, K_i, I_k, h, d_ij_processed, d_kj_processed, 
            p, t, valid_combinations
        )
    else:
        raise ValueError(f"Tipo de problema no reconocido: {problem_type}. Elija 'P1' o 'P2'.")
    
    model_time = time.time() - model_start
    logger.debug(f"Tiempo en crear modelo: {model_time:.4f} segundos")
    logger.debug(f"Variables del modelo: {model.numVariables()}")
    logger.debug(f"Restricciones del modelo: {model.numConstraints()}")
    
    # Resolver
    solve_start = time.time()
    
    # Configurar solver optimizado
    if optimizer == "CBC":
        log_path = f"i{instance}_cbc.log"
        solver = pulp.PULP_CBC_CMD(
            msg=True, 
            timeLimit=time_limit, 
            logPath=os.path.join(log_dir, log_path),
            gapRel=gap_rel,
            threads=0,
            options=[
                'cuts', 'on',
                'heur', 'on',
                'preprocess', 'on',
                'gomory', 'on',
                'knapsack', 'on',
                'probing', 'on',
                'clique', 'on'
            ]
        )
    elif optimizer == "GLPK":
        log_path = f"i{instance}_glpk.log"
        options = ['--log', os.path.join(log_dir, log_path), '--cuts']
        if time_limit:
            options.extend(['--tmlim', str(time_limit)])
        solver = pulp.GLPK_CMD(msg=True, options=options)
    elif optimizer == "CPLEX":
        log_path = f"i{instance}_cplex.log"
        solver = pulp.CPLEX_CMD(
            path="/home/andrea/Documentos/Close-Enough-Facility-Location/cplex/bin/x86-64_linux/cplex",
            msg=True,
            timeLimit=time_limit,
            logPath=os.path.join(log_dir, log_path),
            options=[f'set mip tolerances mipgap {gap_rel}']
        )
    else:
        logger.warning(f"Optimizer {optimizer} not recognized. Using default CBC solver.")
        log_path = f"i{instance}_cbc.log"
        solver = pulp.PULP_CBC_CMD(
            msg=True, 
            timeLimit=time_limit, 
            logPath=os.path.join(log_dir, log_path),
            gapRel=gap_rel
        )
    
    model.solve(solver)
    solve_time = time.time() - solve_start
    
    logger.info(f"Tiempo de resolución: {solve_time:.4f} segundos")
    logger.info(f"Tiempo total: {preprocessing_time + model_time + solve_time:.4f} segundos")
    
    # Extraer variables para guardar resultados
    if problem_type == "P1":
        x, z, y, nu, s = variables['x'], variables['z'], variables['y'], variables['nu'], variables['s']
    else:  # P2
        w, y, nu = variables['w'], variables['y'], variables['nu']
    
    # Guardar resultados (código original adaptado)
    if optimal_solution:
        results_dict = {}
        filename = f"{n_nodos}.json"
        filepath = os.path.join(result_dir, filename)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    results_dict = json.load(f)
            except json.JSONDecodeError:
                print(f"Advertencia: El archivo {filepath} no es un JSON válido o está vacío. Se creará uno nuevo.")
                results_dict = {}
        
        if pulp.value(model.objective):
            objective_value = str(round(pulp.value(model.objective), 2))
        else:
            objective_value = "Infeasible"
        
        key = f"i{instance}"
        results_dict[key] = objective_value
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        print(f"Instancia {instance} ({problem_type}) objetivo {objective_value} guardado en {filepath}")
    
    else:
        # Guardar resultados detallados
        filename = f"i{instance}_{problem_type}_result.txt"
        filepath = os.path.join(result_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Instancia: {instance}\n")
            f.write(f"Tipo de Problema: {problem_type}\n")
            f.write(f"Estado: {pulp.LpStatus[model.status]}\n")
            f.write(f"Valor Objetivo: {str(round(pulp.value(model.objective), 2))}\n")
            f.write(f"Tiempo de Preprocesamiento: {preprocessing_time:.4f} segundos\n")
            f.write(f"Tiempo de Creación de Modelo: {model_time:.4f} segundos\n")
            f.write(f"Tiempo de Resolución: {solve_time:.4f} segundos\n")
            f.write(f"Tiempo Total: {preprocessing_time + model_time + solve_time:.4f} segundos\n")
            f.write(f"Variables del Modelo: {model.numVariables()}\n")
            f.write(f"Restricciones del Modelo: {model.numConstraints()}\n")
            
            if use_preprocessing:
                f.write(f"\nOptimizaciones aplicadas:\n")
                f.write(f"  Combinaciones x válidas: {len(valid_combinations['x_indices'])}\n")
                f.write(f"  Combinaciones z válidas: {len(valid_combinations['z_indices'])}\n")
                f.write(f"  Combinaciones s válidas: {len(valid_combinations['s_indices'])}\n")
                if problem_type == "P2":
                    f.write(f"  Combinaciones w válidas: {len(valid_combinations['w_indices'])}\n")
            
            # Guardar solución si es óptima
            if model.status == pulp.LpStatusOptimal:
                f.write("\nInstalaciones Abiertas (y):\n")
                for j in J:
                    if pulp.value(y[j]) == 1:
                        f.write(f"  Instalación {j}\n")
                
                f.write("Puntos de Recogida Abiertos (nu):\n")
                for k in K:
                    if pulp.value(nu[k]) == 1:
                        f.write(f"  Punto {k}\n")
                
                if problem_type == "P1":
                    f.write("Asignaciones Cliente-Instalación (x):\n")
                    for (i, j) in valid_combinations['x_indices']:
                        if pulp.value(x[(i, j)]) == 1:
                            f.write(f"  Cliente {i} a Instalación {j}\n")
                    
                    f.write("Asignaciones Cliente-PuntoRecogida (z):\n")
                    for (i, k) in valid_combinations['z_indices']:
                        if pulp.value(z[(i, k)]) == 1:
                            f.write(f"  Cliente {i} a Punto {k}\n")
                    
                    f.write("Flujos PuntoRecogida-Instalación (s):\n")
                    for (k, j) in valid_combinations['s_indices']:
                        if pulp.value(s[(k, j)]) > 0:
                            f.write(f"  Flujo desde Punto {k} a Instalación {j}: {pulp.value(s[(k, j)])}\n")
                
                elif problem_type == "P2":
                    f.write("Asignaciones Cliente-PuntoRecogida-Instalación (w):\n")
                    for (i, k_or_i, j) in valid_combinations['w_indices']:
                        if pulp.value(w[(i, k_or_i, j)]) == 1:
                            if k_or_i == i:
                                f.write(f"  Cliente {i} (directo) a Instalación {j}\n")
                            else:
                                f.write(f"  Cliente {i} a Punto {k_or_i} servido por Instalación {j}\n")
        
        print(f"Instancia {instance} ({problem_type}) resuelta. Estado: {pulp.LpStatus[model.status]}")
        print(f"Tiempo total de optimización: {preprocessing_time + model_time + solve_time:.4f} segundos")




# I = [i+1 for i in range(n)] # Usaremos los datos que tengamos almacenados en la instancia i1 en instances/i1.txt
# J = I # En este caso estamos considerando que las instalaciones son los mismos clientes por ello I = J
# # R = 2.98 # Radio de cobertura de los puntos de recogida, a partir de este dato podremos obtener los puntos de recogida
# K = [k+1 for k in range(len(dist_pickuppoints))] # Puntos de recogida, para obtenerlos se debe ejecutar el algoritmo de puntos de recogida
# K_i = get_dict_pickuppoints(I=instance_dict['nodes'], K=dist_pickuppoints, R=R, type="costumer") # Diccionario que para cada cliente i (en I) da el subconjunto Ki (lista de k en K)
# I_k = get_dict_pickuppoints(I=instance_dict['nodes'], K=dist_pickuppoints, R=R, type="candidate") # Diccionario que para cada candidato k (en K) da el subconjunto I_k de clientes que pueden ir a k
# # print(I_k)
# h = [int(value['demand']) for _, value in instance_dict['nodes'].items()]  # demanda de cada cliente, esta info la sacamos de la instancia
# # print("Demanda de cada cliente:", h)
# d_ij = instance_dict['d'] # Diccionario doble para distancia/costo entre i en I y j en J
# # print(d_ij)
# d_kj = get_dict_distances(I=instance_dict['nodes'], K=dist_pickuppoints) # Diccionario doble para distancia/costo entre k en K y j en J
# print(d_kj)
# t = 10
# M_param = sum(h[i] for i in range(n))

