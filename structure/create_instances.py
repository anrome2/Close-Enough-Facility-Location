
import math
import os
import cupy as cp
import numpy as np
from numba import cuda, jit
import itertools

from structure.pickuppoints import clear_gpu_memory
from structure.instances import readInstance
from structure.pickup import get_dict_distances, get_dict_pickuppoints, get_pickuppoints, get_max_from_nested_dict

# Listas de cantidad de nodos
NODOS_50 = [[10, 2, 3], [20, 2, 10], [30, 3, 10], [35, 3, 10], [40, 4, 10], [50, 4, 10]]
NODOS_100 = [[55, 4, 10], [60, 4, 10], [65, 4, 10], [70, 4, 10], [75, 4, 10], [80, 4, 10], [85, 4, 10], [90, 4, 10], [100, 4, 10]]

def read_file(path):
    with open(path, 'r') as archivo:
        lineas = archivo.readlines()
    return int(lineas[0].strip()), lineas

def extract_instance_data(M, lines, instance, n_nodos_total, n_nodos, n_intances, n_facilities):
    indice, num_problema = 1, 0
    instance_dict = {}
    
    while num_problema < M:
        num_problema, best_value = map(int, lines[indice].strip().split())
        n, p, _ = map(int, lines[indice + 1].strip().split())
        
        if (num_problema in [instance, instance + 10]) and (n == n_nodos_total):
            instance_dict["n"], instance_dict["p"], instance_dict["t"] = n_nodos, n_intances, n_facilities
            instance_dict["nodes"] = {}
            
            for i in range(n_nodos):
                data = list(map(int, lines[indice + 2 + i].strip().split()))
                instance_dict["nodes"][data[0]] = tuple(data[1:])
            
            return instance_dict
        
        indice += 2 + n
    return None

def create_instance(instance):
    if not instance:
        return ""
    
    salida = f"{instance['n']} {instance['p']}\n"
    nodes = instance["nodes"]
    
    for i in range(1, instance["n"] + 1):
        x1, y1, d1 = nodes[i]
        salida += f"{i} {x1} {y1} {d1}\n"
        
        for j in range(i + 1, instance["n"] + 1):
            x2, y2, _ = nodes[j]
            dist = round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 3)
            salida += f"{i} {j} {dist}\n"
    
    return salida

def map_R(idx):
    if idx < 16:
        return 0.025
    elif idx < 31:
        return 0.05
    elif idx < 46:
        return 0.10
    elif idx < 61:
        return 0.15
    else:
        return 0

def save_instance(instance, filename):
    if instance:
        with open(filename, "w") as f:
            f.write(instance)

def create_instances(instance=None, n_nodos=None):
    config = {"instance": instance, "n_nodos": n_nodos}
    file_path = 'data/pmedcap1.txt'
    M, lineas = read_file(file_path)
    
    nodos_lista = NODOS_50 if config["n_nodos"] == 50 else NODOS_100
    
    for problema in range(M):
        print(f"Problema {problema + 1}")
        
        for i, (n, p, t) in enumerate(nodos_lista):
            instances_dict = extract_instance_data(
                                M=M, 
                                lines=lineas, 
                                instance=config["instance"], 
                                n_nodos_total=config["n_nodos"], 
                                n_nodos=n, 
                                n_intances=p, 
                                n_facilities=t
                                )
            salida = create_instance(instances_dict)
            os.makedirs(f"./instances/p{instance}", exist_ok=True)
            if salida:
                for m in range(4):
                    instancia_filename = f"./instances/p{instance}/i{(i+1 if config['n_nodos'] == 50 else i+7) + 15*m}.txt"
                    print(f"Guardando instancia {instancia_filename}")
                    save_instance(salida, instancia_filename)

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


if __name__ == "__main__":
    for i in range(1, 11):
        create_instances(i, 50)
        create_instances(i, 100)
        # Limpiar memoria GPU después de cada instancia
        clear_gpu_memory()

