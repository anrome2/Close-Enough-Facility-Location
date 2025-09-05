import itertools

import math
import os
import cupy as cp
import numpy as np
from numba import cuda, jit
import itertools


@cuda.jit
def distance_kernel(x1, y1, x2, y2, distances):
    """Kernel CUDA para calcular distancias euclideas en paralelo"""
    idx = cuda.grid(1)
    if idx < distances.shape[0]:
        dx = x2[idx] - x1[idx]
        dy = y2[idx] - y1[idx]
        distances[idx] = math.sqrt(dx * dx + dy * dy)

def intersection_line_circle_vectorized(centers1, centers2, R):
    """
    Versión vectorizada usando CuPy para calcular intersecciones línea-círculo
    """
    # Convertir a arrays de CuPy
    centers1 = cp.asarray(centers1)
    centers2 = cp.asarray(centers2)
    
    # Calcular diferencias
    dx = centers2[:, 0] - centers1[:, 0]
    dy = centers2[:, 1] - centers1[:, 1]
    
    # Distancias
    d = cp.sqrt(dx**2 + dy**2)
    
    # Evitar división por cero
    valid_mask = d > 1e-10
    
    # Vectores unitarios
    ux = cp.zeros_like(dx)
    uy = cp.zeros_like(dy)
    ux[valid_mask] = dx[valid_mask] / d[valid_mask]
    uy[valid_mask] = dy[valid_mask] / d[valid_mask]
    
    # Puntos de intersección
    p1_x = centers1[:, 0] + R * ux
    p1_y = centers1[:, 1] + R * uy
    p2_x = centers2[:, 0] - R * ux
    p2_y = centers2[:, 1] - R * uy
    
    # Combinar puntos
    points = cp.stack([
        cp.stack([p1_x, p1_y], axis=1),
        cp.stack([p2_x, p2_y], axis=1)
    ], axis=1)
    
    return points[valid_mask].reshape(-1, 2)

def intersection_two_circles_vectorized(centers1, centers2, R):
    """
    Versión vectorizada usando CuPy para calcular intersecciones círculo-círculo
    """
    centers1 = cp.asarray(centers1)
    centers2 = cp.asarray(centers2)
    
    dx = centers2[:, 0] - centers1[:, 0]
    dy = centers2[:, 1] - centers1[:, 1]
    d = cp.sqrt(dx**2 + dy**2)
    
    # Filtrar casos válidos
    valid_mask = (d > 1e-10) & (d < 2 * R)
    
    if not cp.any(valid_mask):
        return cp.array([]).reshape(0, 2)
    
    # Calcular solo para casos válidos
    d_valid = d[valid_mask]
    dx_valid = dx[valid_mask]
    dy_valid = dy[valid_mask]
    centers1_valid = centers1[valid_mask]
    centers2_valid = centers2[valid_mask]
    
    a = d_valid / 2
    h = cp.sqrt(R**2 - a**2)
    
    # Punto medio
    mx = (centers1_valid[:, 0] + centers2_valid[:, 0]) / 2
    my = (centers1_valid[:, 1] + centers2_valid[:, 1]) / 2
    
    # Vector perpendicular unitario
    ux = -dy_valid / d_valid
    uy = dx_valid / d_valid
    
    # Puntos de intersección
    points = cp.stack([
        cp.stack([mx + h * ux, my + h * uy], axis=1),
        cp.stack([mx - h * ux, my - h * uy], axis=1)
    ], axis=1)
    
    return points.reshape(-1, 2)

def get_pickuppoints_optimized(R: float, nodes: dict) -> list:
    """
    Versión optimizada usando CuPy para generar puntos de recogida
    """
    # Convertir nodos a array
    coords = cp.array([(float(node['x']), float(node['y'])) for node in nodes.values()])
    n_nodes = len(coords)
    
    if n_nodes < 2:
        return []
    
    # Generar todas las combinaciones de pares
    pairs_idx = cp.array(list(itertools.combinations(range(n_nodes), 2)))
    centers1 = coords[pairs_idx[:, 0]]
    centers2 = coords[pairs_idx[:, 1]]
    
    # Calcular distancias entre pares
    distances = cp.linalg.norm(centers2 - centers1, axis=1)
    
    # Clasificar pares por tipo de intersección
    type1_mask = distances > 2 * R  # Solo intersección línea-círculo
    type2_mask = distances < R      # Solo intersección círculo-círculo
    type3_mask = (distances >= R) & (distances <= 2 * R)  # Ambos tipos
    
    pickuppoints = set()
    
    # Tipo 1: Intersecciones línea-círculo
    if cp.any(type1_mask | type3_mask):
        mask = type1_mask | type3_mask
        intersections = intersection_line_circle_vectorized(
            centers1[mask], centers2[mask], R
        )
        if len(intersections) > 0:
            intersections_cpu = cp.asnumpy(intersections)
            for point in intersections_cpu:
                pickuppoints.add((round(float(point[0]), 6), round(float(point[1]), 6)))
    
    # Tipo 2: Intersecciones círculo-círculo
    if cp.any(type2_mask | type3_mask):
        mask = type2_mask | type3_mask
        intersections = intersection_two_circles_vectorized(
            centers1[mask], centers2[mask], R
        )
        if len(intersections) > 0:
            intersections_cpu = cp.asnumpy(intersections)
            for point in intersections_cpu:
                pickuppoints.add((round(float(point[0]), 6), round(float(point[1]), 6)))
    
    return list(pickuppoints)

def get_dict_pickuppoints_optimized(customers_dict: dict, pickups_list: list, R: float, type: str = "customer") -> dict:
    """
    Versión optimizada usando CuPy para mapeo de accesibilidad
    """
    customer_coords = cp.array([(float(c['x']), float(c['y'])) for c in customers_dict.values()])
    pickup_coords = cp.array(pickups_list) if pickups_list else cp.array([]).reshape(0, 2)
    num_customers = len(customer_coords)
    
    if len(pickup_coords) == 0:
        return {}
    
    mapping_dict = {}
    
    if type == "customer":
        # Calcular distancias usando broadcasting
        # customer_coords: (n_customers, 2), pickup_coords: (n_pickups, 2)
        customer_expanded = customer_coords[:, cp.newaxis, :]  # (n_customers, 1, 2)
        pickup_expanded = pickup_coords[cp.newaxis, :, :]      # (1, n_pickups, 2)
        
        # Distancias: (n_customers, n_pickups)
        distances = cp.linalg.norm(customer_expanded - pickup_expanded, axis=2)
        accessible_mask = distances <= R
        
        for i in range(num_customers):
            accessible_pickups = cp.where(accessible_mask[i])[0] + num_customers + 1
            mapping_dict[i + 1] = cp.asnumpy(accessible_pickups).tolist()
            
    elif type == "candidate":
        # pickup_coords: (n_pickups, 2), customer_coords: (n_customers, 2)
        pickup_expanded = pickup_coords[:, cp.newaxis, :]     # (n_pickups, 1, 2)
        customer_expanded = customer_coords[cp.newaxis, :, :] # (1, n_customers, 2)
        
        # Distancias: (n_pickups, n_customers)
        distances = cp.linalg.norm(pickup_expanded - customer_expanded, axis=2)
        accessible_mask = distances <= R
        
        for i in range(len(pickup_coords)):
            accessible_customers = cp.where(accessible_mask[i])[0] + 1
            mapping_dict[i + num_customers + 1] = cp.asnumpy(accessible_customers).tolist()
    
    return mapping_dict

def get_dict_distances_optimized(K: list, I: dict) -> dict:
    """
    Versión optimizada usando CuPy para cálculo de distancias
    """
    n = len(I)
    if not K:
        return {}
    
    # Convertir a arrays de CuPy
    pickup_coords = cp.array(K)
    customer_coords = cp.array([(float(node['x']), float(node['y'])) for node in I.values()])
    
    # Calcular todas las distancias de una vez usando broadcasting
    pickup_expanded = pickup_coords[:, cp.newaxis, :]     # (n_pickups, 1, 2)
    customer_expanded = customer_coords[cp.newaxis, :, :] # (1, n_customers, 2)
    
    # Matriz de distancias: (n_pickups, n_customers)
    distance_matrix = cp.linalg.norm(pickup_expanded - customer_expanded, axis=2)
    
    # Convertir a CPU y crear diccionario
    distance_matrix_cpu = cp.asnumpy(distance_matrix)
    dict_distances = {}
    
    for k in range(len(K)):
        dict_distances[k + n + 1] = {}
        for i in range(n):
            dict_distances[k + n + 1][i + 1] = round(float(distance_matrix_cpu[k, i]), 3)
    
    return dict_distances

# ============================================================================
# FUNCIONES AUXILIARES PARA LIBERACIÓN DE MEMORIA
# ============================================================================

def clear_gpu_memory():
    """Limpia la memoria de la GPU"""
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

def get_max_from_nested_dict(d_ij):
    """
    Calcula el valor máximo en un diccionario de diccionarios anidados
    utilizando la GPU a través de cuML.

    Args:
        d_ij (dict): Un diccionario anidado con la estructura {i: {j: valor}}.

    Returns:
        float: El valor máximo encontrado en el diccionario.
    """
    # 1. Extrae todos los valores numéricos del diccionario anidado
    #    y los coloca en una lista de Python.
    flat_values = [value for inner_dict in d_ij.values() for value in inner_dict.values()]

    # 2. Convierte la lista de valores a un array de cuPy.
    #    Esto mueve los datos de la memoria RAM a la memoria de la GPU.
    gpu_array = cp.array(flat_values, dtype=cp.float32)

    # 3. Usa la función de reducción 'max' de cuPy para encontrar
    #    el valor máximo en el array en la GPU.
    max_value = gpu_array.max()

    # 4. Devuelve el resultado como un tipo de dato estándar de Python.
    return max_value.get()