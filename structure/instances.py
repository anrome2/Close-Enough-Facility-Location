def readInstance(path):
    instance = {}
    with open(path, "r") as f:
        # Leer n y p
        n, p = map(int, f.readline().split())
        instance['n'] = n
        instance['p'] = p
        instance['d'] = {}  # Matriz de distancias
        instance['nodes'] = {}  # Información de los nodos
        
        # Leer nodos hasta encontrar una línea con solo 3 columnas (indicando que es una distancia)
        for line in f:
            data = line.split()
            if len(data) == 3:
                u, v, d = data[0], data[1], data[2]
                u = int(u)
                v = int(v)
                d = round(float(d), 3)
                if u not in instance['d']:
                    instance['d'][u] = {}
                if v not in instance['d']:
                    instance['d'][v] = {}
                instance['d'][u][v] = d
            elif len(data) == 4:
                node_id, x, y, demand = data[0], data[1], data[2], data[3]
                instance['nodes'][node_id] = {'x': x, 'y': y, 'demand': demand}

    return instance