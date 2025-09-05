import numpy as np
import itertools
def intersection_line_circle(x1: float, y1: float, x2: float, y2: float, R: float) -> set:
    """
    Dados dos centros de circunferencias y un radio común R, devuelve los puntos de intersección
    entre la línea que une ambos centros y el borde de cada circunferencia.

    Devuelve una lista con dos puntos: uno sobre el borde de la circunferencia centrada en (x1, y1)
    y otro sobre la de (x2, y2), ambos sobre la línea que conecta los centros.
    """
    dx = x2 - x1
    dy = y2 - y1
    # Distancia euclidiana entre dos puntos del espacio
    d = np.hypot(dx, dy)
    if d == 0:
        return None
    # Vector unitario en la dirección de la línea entre centros
    ux = dx / d
    uy = dy / d
    # Punto sobre el borde de la primera circunferencia en dirección al segundo centro
    p1 = (x1 + R * ux, y1 + R * uy)
    # Punto sobre el borde de la segunda circunferencia en dirección opuesta al primer centro
    p2 = (x2 - R * ux, y2 - R * uy)

    return set([p1, p2])
def intersection_two_circles(x1: float, y1: float, x2: float, y2: float, R: float) -> set:
    """
    Calcula los puntos de intersección entre dos circunferencias de radio R
    centradas en (x1, y1) y (x2, y2).
    """
    d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if d == 0:
        return None

    # Calcular los puntos de intersección
    a = d / 2
    h = np.sqrt(R**2 - a**2)

    # Punto medio entre los centros
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2

    # Vector perpendicular unitario
    ux = -(y2 - y1) / d
    # print("UX ", ux)
    uy = (x2 - x1) / d
    # print("UY ", uy)

    # Los dos puntos de intersección
    intersection_points = [
        (mx + h * ux, my + h * uy),
        (mx - h * ux, my - h * uy)
    ]

    return set(intersection_points)
def get_pickuppoints(R: float, nodes: dict) -> list:
    """
    Genera los puntos de recogida K a partir de los nodos dados según la teoría CEFL.

    Parámetros:
    - R: Radio fijo para todas las circunferencias
    - nodes: Diccionario de nodos con formato {id: {'x': x_coord, 'y': y_coord}}

    Retorna:
    - Lista de puntos de recogida K únicos
    """
    pickuppoints = set()
    nodes_list = []

    # Convertir diccionario a lista de coordenadas
    for node_dict in nodes.values():
        x = float(node_dict['x'])
        y = float(node_dict['y'])
        nodes_list.append((x, y))

    # Iterar sobre todos los pares de nodos
    for (x1, y1), (x2, y2) in itertools.combinations(nodes_list, 2):
        d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        # print("DIST ", d)
        # Solo procesar si la distancia permite intersecciones útiles
        if d > 2 * R:
            # print("Tipo 1: Intersecciones de la recta con cada circunferencia")
            # Tipo 1: Intersecciones de la recta con cada circunferencia
            intersections_line = intersection_line_circle(x1, y1, x2, y2, R)
            if intersections_line:
            # print("INTERSECTIONS LINE ", intersections_line)
                pickuppoints.update(intersections_line)
        elif d < R:
            # Tipo 2: Intersecciones entre las dos circunferencias
            # print("Tipo 2: Intersecciones entre las dos circunferencias")
            intersections_circles = intersection_two_circles(x1, y1, x2, y2, R)
            # print("INTERSECTIONS CIRCLE ", intersections_circles)
            if intersections_circles:
                pickuppoints.update(intersections_circles)
        elif d < 2 * R and d > R:
            # print("Tipo 1: Intersecciones de la recta con cada circunferencia")
            # Tipo 1: Intersecciones de la recta con cada circunferencia
            intersections_line = intersection_line_circle(x1, y1, x2, y2, R)
            # print("INTERSECTIONS LINE ", intersections_line)
            if intersections_line:
                pickuppoints.update(intersections_line)
            # Tipo 2: Intersecciones entre las dos circunferencias
            # print("Tipo 2: Intersecciones entre las dos circunferencias")
            intersections_circles = intersection_two_circles(x1, y1, x2, y2, R)
            # print("INTERSECTIONS CIRCLE ", intersections_circles)
            if intersections_circles:
                pickuppoints.update(intersections_circles)
    # Convertir set a lista y redondear para evitar problemas de precisión
    return [(round(x, 6), round(y, 6)) for x, y in pickuppoints]
def get_dict_pickuppoints(customers_dict: dict, pickups_list: list, R: float, type: str = "customer") -> dict:
    """
    Crea un diccionario de accesibilidad para clientes o puntos de recogida.
    Parámetros:
    - customers_dict: Un diccionario de clientes, ej: {1: {'x': 10, 'y': 20}, ...}
    - pickups_list: Una lista de puntos de recogida, ej: [(x1, y1), (x2, y2), ...]
    - R: Radio máximo de alcance para un punto de recogida.
    - type: 'customer' para mapear clientes a pickups, 'candidate' para mapear pickups a clientes.
    Retorna:
    - Un diccionario de mapeo de accesibilidad.
    """

    # Extraemos las coordenadas de los clientes en una lista de tuplas
    customer_coords = [(float(c['x']), float(c['y'])) for c in customers_dict.values()]
    num_customers = len(customer_coords)

    if type == "customer":
        mapping_dict = {}
        for i, (xi, yi) in enumerate(customer_coords):
            # print("CLIENTE ", i)
            accessible_pickups = []
            for k, (xk, yk) in enumerate(pickups_list):
                distance = np.linalg.norm(np.array([xi, yi]) - np.array([xk, yk]))
                # print("DISTANCE ", round(float(distance), 3))
                if (round(float(distance), 2)) <= R:
                    accessible_pickups.append(k + num_customers + 1)
            mapping_dict[i + 1] = accessible_pickups

    elif type == "candidate":
        mapping_dict = {}
        for i, (xi, yi) in enumerate(pickups_list):
            accessible_customers = []
            for k, (xk, yk) in enumerate(customer_coords):
                distance = np.linalg.norm(np.array([xi, yi]) - np.array([xk, yk]))
                if (round(float(distance), 2)) <= R:
                    accessible_customers.append(k + 1)
            mapping_dict[i + num_customers + 1] = accessible_customers

    else:
        raise ValueError("El tipo debe ser 'customer' o 'candidate'")
    return mapping_dict
def get_dict_distances(K: list, I: dict) -> dict:
    """
    Crea un diccionario de distancias entre puntos de recogida y clientes.

    Parámetros:
    - K: Lista de puntos de recogida [(x, y), ...]
    - I: Diccionario de clientes {id: {'x': x_coord, 'y': y_coord}}

    Retorna:
    - Diccionario {pickup_id: {customer_id: distance}}
    """
    n = len(I)
    dict_distances = {}

    # Convertir diccionario de clientes a lista de coordenadas
    customer_coords = [(float(node['x']), float(node['y'])) for node in I.values()]

    for k, (xk, yk) in enumerate(K):
        dict_distances[k + n + 1] = {}
        for i, (xi, yi) in enumerate(customer_coords):
            distance = np.sqrt((float(xk) - xi)**2 + (float(yk) - yi)**2)
            dict_distances[k + n + 1][i + 1] = round(distance, 3)

    return dict_distances

import numpy as np

def get_max_from_nested_dict(d_ij):
    """
    Calcula el valor máximo en un diccionario de diccionarios anidados
    """
    flat_values = [value for inner_dict in d_ij.values() for value in inner_dict.values()]
    return np.max(flat_values)