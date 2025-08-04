import os
import time
import pulp

def milp_ceflp(params, instance, result_dir, problem_type="P1", optimizer="CBC", time_limit=None):
    """
    Resuelve el problema de localización de instalaciones Close-Enough (CEFLP)
    utilizando formulaciones de Programación Lineal Entera Mixta (MILP) con PuLP.

    Soporta dos formulaciones: P1 (Two-index formulation) y P2 (Three-index formulation).

    Args:
        params (dict): Diccionario de parámetros del problema, incluyendo:
            - I (list): Conjunto de índices de clientes.
            - J (list): Conjunto de índices de instalaciones potenciales.
            - K (list): Conjunto de índices de puntos de recogida potenciales.
            - h (list): Demandas de los clientes.
            - d_ij (dict): Distancias entre clientes (o instalaciones) y instalaciones.
            - d_kj (dict): Distancias entre puntos de recogida (o clientes) y instalaciones.
            - K_i (dict): Conjunto de puntos de recogida válidos para cada cliente i.
            - I_k (dict): Conjunto de clientes que pueden usar cada punto de recogida k.
            - p (int): Número de instalaciones a abrir.
            - t (int): Número de puntos de recogida a abrir.
            - M_param (float): Gran número M para linealización (solo para P1).
        instance (int): Índice de la instancia (para nombrar el archivo de resultados).
        result_dir (str): Directorio para guardar los resultados.
        problem_type (str): Tipo de formulación a resolver ("P1" o "P2"). Por defecto "P1".
        optimizer (str): Optimizador a usar ("CBC", "GLPK", "CPLEX"). Por defecto "CBC".
        time_limit (int, optional): Límite de tiempo para el optimizador en segundos. Por defecto None.
    """
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

    # Crear modelo
    model = pulp.LpProblem(f"{problem_type}_CEFLP", pulp.LpMinimize)

    # Variables y Función Objetivo basadas en el tipo de problema
    if problem_type == "P1":
        M_param = params['M_param'] # M_param solo es necesario para P1
        # Variables P1
        x = pulp.LpVariable.dicts("x", (I, J), cat=pulp.LpBinary)
        z = pulp.LpVariable.dicts("z", (I, K), cat=pulp.LpBinary)
        y = pulp.LpVariable.dicts("y", J, cat=pulp.LpBinary)
        nu = pulp.LpVariable.dicts("nu", K, cat=pulp.LpBinary)
        s = pulp.LpVariable.dicts("s", (K, J), lowBound=0, cat=pulp.LpContinuous)

        # Función objetivo P1: Minimizar el costo de asignación directa y el costo de distribución vía puntos de recogida
        # Nota: La formulación del paper usa d_ij[i][j] o d_ij[j][i] dependiendo de la relación i < j o i > j.
        # Aquí asumimos que d_ij es simétrica o ya está preprocesada para que d_ij[i][j] sea la distancia correcta.
        # Se ajusta la suma de h_i * d_ij para clientes a instalaciones y d_kj * s para recogida a instalación
        model += (
            pulp.lpSum([h[i-1] * d_ij[min(i,j)][max(i,j)] * x[i][j] if i !=j  else 0 for i in I for j in J])
            + pulp.lpSum([d_kj[k][j] * s[k][j] for k in K for j in J])
        )

        # Restricciones P1
        # Restricción (3) del paper: Se abren exactamente p instalaciones
        model += pulp.lpSum([y[j] for j in J]) == p
        # Restricción (4) del paper: Se abren exactamente t puntos de recogida
        model += pulp.lpSum([nu[k] for k in K]) == t
        # Restricción (5) del paper: Cada cliente se asigna ya sea a una instalación o a un punto de recogida (dentro de su conjunto Ki)
        for i in I:
            model += (
                pulp.lpSum([x[i][j] for j in J])
                + pulp.lpSum([z[i][k] for k in K_i[i]])
                == 1
            )
        # Restricción (6) del paper: Para cada candidato k, la suma de los envíos desde k a todas las instalaciones
        # debe ser igual a la demanda de los clientes que se asignan a k.
        for k in K:
            model += (
                pulp.lpSum([s[k][j] for j in J])
                == pulp.lpSum([h[i-1] * z[i][k] for i in I_k[k]])
            )
        # Restricción (7) del paper: Un envío solo se puede realizar si la instalación está abierta
        for j in J:
            model += pulp.lpSum([s[k][j] for k in K]) <= M_param * y[j]
        # Restricción (8) del paper: Solo se puede asignar un cliente a un punto de recogida k si éste está abierto
        for i in I:
            for k in K_i[i]:
                model += z[i][k] <= nu[k]
        # Restricción (9) del paper: Un cliente solo puede asignarse a una instalación j si ésta está abierta
        for i in I:
            for j in J:
                model += x[i][j] <= y[j]

    elif problem_type == "P2":
        # Variables P2
        # W_ikj: 1 si el cliente i es asignado al punto de recogida k (o directamente a sí mismo si k=i)
        # y este es servido por la instalación j
        w = pulp.LpVariable.dicts("w", (I, K + [i for i in I], J), cat=pulp.LpBinary) # K_i U {i} es el conjunto
        y = pulp.LpVariable.dicts("y", J, cat=pulp.LpBinary)
        nu = pulp.LpVariable.dicts("nu", K, cat=pulp.LpBinary) # nu_k para k en K

        # Función objetivo P2: Minimizar el costo total
        # Suma h_i * d_kj * w_ikj. d_kj debe ser la distancia correcta.
        # Si k=i, d_kj se interpreta como d_ij (distancia cliente-instalación)
        # Si k!=i, d_kj se interpreta como d_kj (distancia punto_recogida-instalación)
        model += pulp.lpSum([
            h[i-1] * (d_ij[min(i,j)][max(i,j)] if k == i and i!=j else (d_kj[k][j] if i!=j else 0)) * w[i][k][j]
            for i in I for k_or_i in K_i[i] + [i] for k in (k_or_i,) # Iterate over K_i[i] and also {i}
            for j in J
            if (k == i and i in J) or (k != i) # Ensure d_ij is defined for direct assignment if i is also a potential facility
                                              # and d_kj is defined for pickup points.
        ])


        # Restricciones P2 (referencia a las ecuaciones del paper)
        # Restricción (19): Se abren exactamente p instalaciones
        model += pulp.lpSum([y[j] for j in J]) == p
        # Restricción (20): Se abren exactamente t puntos de recogida
        model += pulp.lpSum([nu[k] for k in K]) == t
        # Restricción (21): Cada cliente debe ser asignado a exactamente una combinación (k, j)
        for i in I:
            model += pulp.lpSum([w[i][k_or_i][j]
                                 for k_or_i in K_i[i] + [i]
                                 for j in J]) == 1
        # Restricción (22): w_ikj <= y_j (Si hay asignación a j, j debe estar abierta)
        for i in I:
            for k_or_i in K_i[i] + [i]:
                for j in J:
                    model += w[i][k_or_i][j] <= y[j]
        # Restricción (23): w_ikj <= nu_k (Si hay asignación a k (donde k!=i), k debe estar abierto)
        for i in I:
            for k in K_i[i]: # K_i only contains actual pickup points (k != i)
                for j in J:
                    model += w[i][k][j] <= nu[k]

    else:
        raise ValueError(f"Tipo de problema no reconocido: {problem_type}. Elija 'P1' o 'P2'.")

    start_time = time.time()
    # Resolver
    if optimizer == "CBC":
        log_path = f"i{instance+1}_cbc.log"
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=time_limit, logPath=os.path.join(result_dir, log_path))
        model.solve(solver)
        end_time = time.time()
    elif optimizer == "GLPK":
        log_path = f"i{instance+1}_glpk.log"
        options = ['--log', os.path.join(result_dir, log_path)]
        solver = pulp.GLPK_CMD(path="./winglpk-4.65/glpk-4.65/w64/glpsol.exe", 
                               msg=True, 
                               options=options,
                               timeLimit=time_limit
                               )
        model.solve(solver)
        end_time = time.time()
    elif optimizer == "CPLEX":
        log_path = f"i{instance+1}_cplex.log"
        solver = pulp.CPLEX_CMD(path="C:/Program Files/IBM/ILOG/CPLEX_Studio2212/cplex/bin/x64_win64/cplex.exe", 
                                msg=True, 
                                timeLimit=time_limit,
                                logPath=os.path.join(result_dir, log_path)
                                )
        model.solve(solver)
        end_time = time.time()
    else:
        print(f"Optimizer {optimizer} not recognized. Using default CBC solver.")
        log_path = f"i{instance+1}_cbc.log"
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=time_limit, logPath=os.path.join(result_dir, log_path))
        model.solve(solver)
        end_time = time.time()

    solve_time = end_time - start_time

    # Guardar resultados
    filename = f"i{instance+1}_{problem_type}_result.txt"
    filepath = os.path.join(result_dir, filename)


    with open(filepath, "w") as f:
        f.write(f"Instancia: {instance+1}\n")
        f.write(f"Tipo de Problema: {problem_type}\n")
        f.write(f"Estado: {pulp.LpStatus[model.status]}\n")
        f.write(f"Valor Objetivo: {str(round(pulp.value(model.objective), 2))}\n")
        f.write(f"Tiempo de Ejecución: {solve_time:.4f} segundos\n")

        if problem_type == "P1" or problem_type == "P2": # Both P1 and P2 use y and nu
            f.write("Instalaciones Abiertas (y):\n")
            for j in J:
                if pulp.value(y[j]) == 1:
                    f.write(f"  Instalación {j}\n")
            f.write("Puntos de Recogida Abiertos (nu):\n")
            for k in K:
                if pulp.value(nu[k]) == 1:
                    f.write(f"  Punto {k}\n")
            if problem_type == "P1":
                f.write("Asignaciones Cliente-Instalación (x):\n")
                for i in I:
                    for j in J:
                        if pulp.value(x[i][j]) == 1:
                            f.write(f"  Cliente {i} a Instalación {j}\n")
                f.write("Asignaciones Cliente-PuntoRecogida (z):\n")
                for i in I:
                    for k in K:
                        if pulp.value(z[i][k]) == 1:
                            f.write(f"  Cliente {i} a Punto {k}\n")
                f.write("Flujos PuntoRecogida-Instalación (s):\n")
                for k in K:
                    for j in J:
                        if pulp.value(s[k][j]) > 0:
                            f.write(f"  Flujo desde Punto {k} a Instalación {j}: {pulp.value(s[k][j])}\n")
            elif problem_type == "P2":
                f.write("Asignaciones Cliente-PuntoRecogida-Instalación (w):\n")
                for i in I:
                    for k_or_i in K_i[i] + [i]:
                        for j in J:
                            if pulp.value(w[i][k_or_i][j]) == 1:
                                if k_or_i == i:
                                    f.write(f"  Cliente {i} (directo) a Instalación {j}\n")
                                else:
                                    f.write(f"  Cliente {i} a Punto {k_or_i} servido por Instalación {j}\n")


    print(f"Instancia {instance+1} ({problem_type}) resuelta. Estado: {pulp.LpStatus[model.status]}")





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

