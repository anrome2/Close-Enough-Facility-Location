import math

# Listas de cantidad de nodos
NODOS_50 = [[10, 2], [20, 2], [30, 3], [35, 3], [40, 4],  [50, 4]]
NODOS_100 = [[55, 4], [60, 4], [65, 4], [70, 4], [75, 4], [80, 4], [85, 4], [90, 4], [100, 4]]

def read_file(path):
    with open(path, 'r') as archivo:
        lineas = archivo.readlines()
    return int(lineas[0].strip()), lineas

def extract_instance_data(M, lines, instance, n_nodos_total, n_nodos, n_intances):
    indice, num_problema = 1, 0
    instance_dict = {}
    
    while num_problema < M:
        num_problema, best_value = map(int, lines[indice].strip().split())
        n, p, _ = map(int, lines[indice + 1].strip().split())
        
        if (num_problema in [instance, instance + 10]) and (n == n_nodos_total):
            instance_dict["n"], instance_dict["p"] = n_nodos, n_intances
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

def save_instance(instance, filename):
    if instance:
        with open(filename, "w") as f:
            f.write(instance)

def create_instances():
    config = {"instance": 1, "n_nodos": 50}
    file_path = 'data/pmedcap1.txt'
    M, lineas = read_file(file_path)
    
    nodos_lista = NODOS_50 if config["n_nodos"] == 50 else NODOS_100
    
    for problema in range(M):
        print(f"Problema {problema + 1}")
        
        for i, (n, p) in enumerate(nodos_lista):
            instances_dict = extract_instance_data(M, lineas, config["instance"], config["n_nodos"], n, p)
            salida = create_instance(instances_dict)
            
            if salida:
                for m in range(4):
                    instancia_filename = f"./instances/i{(i+1 if config['n_nodos'] == 50 else i+7) + 15*m}.txt"
                    print(f"Guardando instancia {instancia_filename}")
                    save_instance(salida, instancia_filename)

if __name__ == "__main__":
    create_instances()

