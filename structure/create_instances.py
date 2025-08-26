
import math
import os
import cupy as cp
import numpy as np
from numba import cuda, jit
import itertools

from structure.pickuppoints import clear_gpu_memory

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

if __name__ == "__main__":
    for i in range(1, 11):
        create_instances(i, 50)
        create_instances(i, 100)
        # Limpiar memoria GPU después de cada instancia
        clear_gpu_memory()

