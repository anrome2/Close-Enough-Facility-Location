import random
import traceback
import numpy as np
from sklearn.cluster import KMeans

class Solution:
    def __init__(self, algorithm, other_solution=None):
        self.algorithm = algorithm
        self.problem = algorithm.problem
        self.cost = float('inf')
        self.time = 0
        self.client_assignments = {}
        
        if other_solution is None:
            self._initialize_empty()
        else:
            self._clone_solution(other_solution)
    
    @property
    def open_facilities(self):
        return [j+1 for j, val in enumerate(self.y) if val == 1]
    
    @property
    def closed_facilities(self):
        return [j+1 for j, val in enumerate(self.y) if val == 0]
    
    @property
    def open_pickups(self):
        return [k+1+self.algorithm.n_customers for k, val in enumerate(self.nu) if val == 1]
    
    @property
    def closed_pickups(self):
        return [k+1+self.algorithm.n_customers for k, val in enumerate(self.nu) if val == 0]
    
    def _initialize_empty(self):
        """Inicialización optimizada según el tipo de problema"""
        self.y = np.zeros(self.algorithm.n_customers, dtype=int)
        self.nu = np.zeros(self.algorithm.n_pickups, dtype=int)
        
        if self.problem == "P1":
            # Variables x_ij: 1 si cliente i es asignado directamente a facilidad j
            self.x = {(i, j): 0 for i in self.algorithm.I for j in self.algorithm.J}
            # Variables z_ik: 1 si cliente i va al punto de recogida k
            self.z = {(i, k): 0 for i in self.algorithm.I for k in self.algorithm.K_i[i]}
            # Variables s_kj: flujo de demanda del punto k a la facilidad j
            self.s = {(k, j): 0.0 for k in self.algorithm.K for j in self.algorithm.J}
        else:  # P2 o default
            self.w = {}
    
    def _clone_solution(self, other):
        """Clonación optimizada usando un solo método"""
        self.y = other.y.copy()
        self.nu = other.nu.copy()
        self.cost = other.cost
        self.client_assignments = other.client_assignments.copy()
        
        if self.problem == "P1":
            self.x = other.x.copy()
            self.z = other.z.copy()
            self.s = other.s.copy()
        else:  # P2 o default
            self.w = other.w.copy()
    
    def _get_distance_ij(self, i, j):
        """Método auxiliar optimizado para obtener distancia entre cliente i e instalación j"""
        if i == j:
            return 0
        elif i < j:
            return self.algorithm.d_ij[i][j]
        else:
            return self.algorithm.d_ij[j][i]
    
    def initialize_greedy(self, alpha=0.3):
        """
        Construcción greedy aleatorizada según GRASP para Close Enough Facility Location
        
        El algoritmo evalúa el costo incremental de añadir cada elemento candidato
        y construye la lista RCL basada en estos costos incrementales.
        """
        try:
            # Reset de variables de decisión
            self._initialize_empty()
            
            # Elementos ya seleccionados
            selected_facilities = set()
            selected_pickups = set()
            
            # Construcción iterativa: alternar entre instalaciones y puntos de recogida
            total_elements = self.algorithm.p + self.algorithm.t
            facilities_needed = self.algorithm.p
            pickups_needed = self.algorithm.t
            
            for iteration in range(total_elements):
                # Determinar qué tipo de elemento añadir basado en lo que falta
                can_add_facility = len(selected_facilities) < facilities_needed
                can_add_pickup = len(selected_pickups) < pickups_needed
                
                if can_add_facility and can_add_pickup:
                    # Si podemos añadir ambos, alternar o elegir basado en beneficio
                    # Por simplicidad, alternamos: facilidades primero
                    add_facility = iteration % 2 == 0
                elif can_add_facility:
                    add_facility = True
                elif can_add_pickup:
                    add_facility = False
                else:
                    break
                
                if add_facility:
                    best_facility = self._select_best_facility_greedy(selected_facilities, selected_pickups, alpha)
                    if best_facility is not None:
                        selected_facilities.add(best_facility)
                        self.y[best_facility-1] = 1
                        self.algorithm.logger.debug(f"Seleccionada facilidad {best_facility}")
                else:
                    best_pickup = self._select_best_pickup_greedy(selected_facilities, selected_pickups, alpha)
                    if best_pickup is not None:
                        selected_pickups.add(best_pickup)
                        self.nu[best_pickup-self.algorithm.n_customers-1] = 1
                        self.algorithm.logger.debug(f"Seleccionado punto de recogida {best_pickup}")
            
            self.algorithm.logger.info(f"Construcción greedy completada:")
            self.algorithm.logger.info(f"  Instalaciones: {sorted(selected_facilities)}")
            self.algorithm.logger.info(f"  Puntos de recogida: {sorted(selected_pickups)}")
            
            # Asignación final de clientes
            self._assign_customers()
            
        except Exception as e:
            self.algorithm.logger.error(f"Error en initialize_greedy: {e}")
            self.algorithm.logger.error(traceback.format_exc())
            raise

    def _select_best_facility_greedy(self, selected_facilities, selected_pickups, alpha):
        """
        Selecciona la mejor facilidad usando criterio greedy con RCL
        """
        candidate_facilities = [j for j in self.algorithm.J if j not in selected_facilities]
        if not candidate_facilities:
            return None
        
        # Calcular costo incremental para cada facilidad candidata
        incremental_costs = []
        
        for j_candidate in candidate_facilities:
            # Evaluar el costo de añadir esta facilidad
            cost_without = self._evaluate_partial_solution_cost(selected_facilities, selected_pickups)
            cost_with = self._evaluate_partial_solution_cost(selected_facilities | {j_candidate}, selected_pickups)
            incremental_cost = cost_with - cost_without
            
            incremental_costs.append((j_candidate, incremental_cost))
            self.algorithm.logger.debug(f"Facilidad {j_candidate}: costo incremental = {incremental_cost}")
        
        # Ordenar por costo incremental (menor es mejor)
        incremental_costs.sort(key=lambda x: x[1])
        
        # Construir RCL
        if not incremental_costs:
            return None
        
        c_min = incremental_costs[0][1]
        c_max = incremental_costs[-1][1]
        
        # Evitar división por cero
        if c_max == c_min:
            threshold = c_min
        else:
            threshold = c_min + alpha * (c_max - c_min)
        
        rcl = [j for j, cost in incremental_costs if cost <= threshold]
        
        self.algorithm.logger.debug(f"RCL facilidades: {rcl} (threshold: {threshold:.2f})")
        
        return random.choice(rcl)

    def _select_best_pickup_greedy(self, selected_facilities, selected_pickups, alpha):
        """
        Selecciona el mejor punto de recogida usando criterio greedy con RCL
        """
        candidate_pickups = [k for k in self.algorithm.K if k not in selected_pickups]
        if not candidate_pickups:
            return None
        
        # Calcular costo incremental para cada punto de recogida candidato
        incremental_costs = []
        
        for k_candidate in candidate_pickups:
            # Evaluar el costo de añadir este punto de recogida
            cost_without = self._evaluate_partial_solution_cost(selected_facilities, selected_pickups)
            cost_with = self._evaluate_partial_solution_cost(selected_facilities, selected_pickups | {k_candidate})
            incremental_cost = cost_with - cost_without
            
            incremental_costs.append((k_candidate, incremental_cost))
            self.algorithm.logger.debug(f"Punto recogida {k_candidate}: costo incremental = {incremental_cost}")
        
        # Ordenar por costo incremental (menor es mejor)
        incremental_costs.sort(key=lambda x: x[1])
        
        # Construir RCL
        if not incremental_costs:
            return None
        
        c_min = incremental_costs[0][1]
        c_max = incremental_costs[-1][1]
        
        # Evitar división por cero
        if c_max == c_min:
            threshold = c_min
        else:
            threshold = c_min + alpha * (c_max - c_min)
        
        rcl = [k for k, cost in incremental_costs if cost <= threshold]
        
        self.algorithm.logger.debug(f"RCL puntos recogida: {rcl} (threshold: {threshold:.2f})")
        
        return random.choice(rcl)
    
    def initialize_kmeans(self):
        """Inicialización usando K-means para una mejor distribución espacial"""
        try:
            self._initialize_facilities_kmeans()
            self._initialize_pickups_kmeans()
            self.evaluate()
            
        except Exception as e:
            self.algorithm.logger.error(f"Error en inicialización K-means: {e}")
            self.initialize_greedy()

    def _initialize_facilities_kmeans(self):
        """Inicialización de instalaciones usando K-means"""
        
        # Crear coordenadas basadas en posición geográfica real si está disponible
        if hasattr(self.algorithm, 'facility_coords') and self.algorithm.facility_coords is not None:
            facility_coords = np.array([coords for coords in self.algorithm.facility_coords])
        else:
            # Fallback: usar distancias promedio como proxy de ubicación
            facility_coords = []
            for j in self.algorithm.J:
                # Calcular centroide de distancias a todos los clientes
                distances = [self._get_distance_ij(i, j) for i in self.algorithm.I]
                avg_dist = np.mean(distances)
                std_dist = np.std(distances)
                facility_coords.append([avg_dist, std_dist])
            facility_coords = np.array(facility_coords)
        
        # Normalizar coordenadas para mejorar clustering
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        facility_coords_norm = scaler.fit_transform(facility_coords)
        
        # Aplicar K-means con múltiples intentos para mejor resultado
        best_score = float('inf')
        best_facilities = None
        
        for attempt in range(5):  # Múltiples intentos con diferentes seeds
            kmeans = KMeans(
                n_clusters=self.algorithm.p, 
                random_state=42 + attempt,
                n_init=20,  # Más inicializaciones para mejor convergencia
                max_iter=300
            )
            cluster_labels = kmeans.fit_predict(facility_coords_norm)
            centers = kmeans.cluster_centers_
            
            # Seleccionar la instalación más cercana a cada centroide
            selected_facilities = set()
            for cluster_id in range(self.algorithm.p):
                cluster_mask = cluster_labels == cluster_id
                if not np.any(cluster_mask):
                    continue
                    
                cluster_coords = facility_coords_norm[cluster_mask]
                cluster_facilities = np.array(list(self.algorithm.J))[cluster_mask]
                
                # Encontrar la instalación más cercana al centroide del cluster
                distances_to_center = np.linalg.norm(
                    cluster_coords - centers[cluster_id], axis=1
                )
                closest_idx = np.argmin(distances_to_center)
                selected_facilities.add(cluster_facilities[closest_idx])
            
            # Evaluar calidad de la solución (suma de distancias promedio)
            if len(selected_facilities) == self.algorithm.p:
                score = sum(
                    np.mean([self._get_distance_ij(i, j) for i in self.algorithm.I])
                    for j in selected_facilities
                )
                if score < best_score:
                    best_score = score
                    best_facilities = selected_facilities.copy()
        
        # Usar la mejor solución encontrada
        selected_facilities = best_facilities or selected_facilities
        
        # Completar si faltan instalaciones (por clusters vacíos)
        if len(selected_facilities) < self.algorithm.p:
            remaining = [j for j in self.algorithm.J if j not in selected_facilities]
            # Ordenar por distancia promedio a clientes
            remaining_scores = {
                j: np.mean([self._get_distance_ij(i, j) for i in self.algorithm.I])
                for j in remaining
            }
            sorted_remaining = sorted(remaining_scores.items(), key=lambda x: x[1])
            needed = self.algorithm.p - len(selected_facilities)
            selected_facilities.update([j for j, _ in sorted_remaining[:needed]])
        
        # Activar instalaciones seleccionadas
        for j in selected_facilities:
            self.y[j] = 1
            
        self.algorithm.logger.info(
            f"[K-means] Instalaciones seleccionadas: {sorted(selected_facilities)}"
        )

    def _initialize_pickups_kmeans(self):
        """Inicialización de puntos de recogida usando K-means"""
        # Obtener instalaciones abiertas
        open_facilities = [j for j in self.algorithm.J if self.y[j] == 1]
        
        if not open_facilities:
            self.algorithm.logger.warning("No hay instalaciones abiertas para inicializar puntos de recogida")
            return
        
        if len(self.algorithm.K) <= self.algorithm.t:
            # Si tenemos pocos puntos de recogida, seleccionar todos
            for k in self.algorithm.K:
                self.nu[k] = 1
            self.algorithm.logger.info(f"[K-means] Todos los puntos de recogida seleccionados: {list(self.algorithm.K)}")
            return
        
        # Crear coordenadas para puntos de recogida
        if hasattr(self.algorithm, 'pickup_coords') and self.algorithm.pickup_coords is not None:
            pickup_coords = np.array([self.algorithm.pickup_coords[k] for k in self.algorithm.K])
        else:
            # Fallback: usar distancias a instalaciones abiertas
            pickup_coords = []
            for k in self.algorithm.K:
                distances = []
                for j in open_facilities:
                    if (k in self.algorithm.d_kj and j in self.algorithm.d_kj[k] and 
                        self.algorithm.d_kj[k][j] != float('inf')):
                        distances.append(self.algorithm.d_kj[k][j])
                
                if distances:
                    avg_dist = np.mean(distances)
                    min_dist = np.min(distances)
                    pickup_coords.append([avg_dist, min_dist])
                else:
                    pickup_coords.append([float('inf'), float('inf')])
            
            pickup_coords = np.array(pickup_coords)
            # Filtrar puntos con distancias infinitas
            valid_mask = ~np.isinf(pickup_coords).any(axis=1)
            if not np.any(valid_mask):
                # Si todos tienen distancias infinitas, usar inicialización greedy
                self.algorithm.logger.warning("Todas las distancias son infinitas, usando inicialización greedy para pickups")
                return
        
        # Normalizar coordenadas
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        pickup_coords_norm = scaler.fit_transform(pickup_coords)
        
        # Aplicar K-means
        kmeans = KMeans(
            n_clusters=min(self.algorithm.t, len(self.algorithm.K)),
            random_state=42,
            n_init=20,
            max_iter=300
        )
        cluster_labels = kmeans.fit_predict(pickup_coords_norm)
        centers = kmeans.cluster_centers_
        
        # Seleccionar el punto de recogida más cercano a cada centroide
        selected_pickups = set()
        for cluster_id in range(min(self.algorithm.t, len(self.algorithm.K))):
            cluster_mask = cluster_labels == cluster_id
            if not np.any(cluster_mask):
                continue
                
            cluster_coords = pickup_coords_norm[cluster_mask]
            cluster_pickups = np.array(list(self.algorithm.K))[cluster_mask]
            
            distances_to_center = np.linalg.norm(
                cluster_coords - centers[cluster_id], axis=1
            )
            closest_idx = np.argmin(distances_to_center)
            selected_pickups.add(cluster_pickups[closest_idx])
        
        # Completar si faltan puntos de recogida
        if len(selected_pickups) < self.algorithm.t:
            remaining = [k for k in self.algorithm.K if k not in selected_pickups]
            # Ordenar por calidad (distancia promedio a instalaciones abiertas)
            remaining_scores = {}
            for k in remaining:
                distances = []
                for j in open_facilities:
                    if (k in self.algorithm.d_kj and j in self.algorithm.d_kj[k] and 
                        self.algorithm.d_kj[k][j] != float('inf')):
                        distances.append(self.algorithm.d_kj[k][j])
                remaining_scores[k] = np.mean(distances) if distances else float('inf')
            
            sorted_remaining = sorted(remaining_scores.items(), key=lambda x: x[1])
            needed = self.algorithm.t - len(selected_pickups)
            selected_pickups.update([k for k, _ in sorted_remaining[:needed]])
        
        # Activar puntos de recogida seleccionados
        for k in selected_pickups:
            self.nu[k] = 1
            
        self.algorithm.logger.info(
            f"[K-means] Puntos de recogida seleccionados: {sorted(selected_pickups)}"
        )
    
    def initialize_random(self):
        try:
            for j in random.sample(self.algorithm.J, self.algorithm.p):
                self.y[j-1] = 1
            self.algorithm.logger.debug(f"Instalaciones seleccionadas: {[j for j in self.algorithm.J if self.y[j-1] == 1]}")
            
            for k in random.sample(self.algorithm.K, self.algorithm.t):
                self.nu[k-self.algorithm.n_customers-1] = 1
            self.algorithm.logger.debug(f"Puntos de recogida seleccionados: {[k for k in self.algorithm.K if self.nu[k-self.algorithm.n_customers-1] == 1]}")
        except Exception as e:
            self.algorithm.logger.error(f"Error en initialize_random: {e}")
            self.algorithm.logger.error(traceback.format_exc())
            raise

    def _evaluate_partial_solution_cost(self, facilities, pickups):
        """
        Evalúa el costo de una solución parcial dados los conjuntos de facilidades y puntos de recogida
        """
        try:
            if not facilities and not pickups:
                return float('inf')  # Solución vacía tiene costo infinito
            
            total_cost = 0
            
            # Para cada cliente, encontrar la opción más barata disponible
            for i in self.algorithm.I:
                client_cost = float('inf')
                demand = self.algorithm.h[i-1]
                
                # Opción 1: Asignación directa a facilidades disponibles
                for j in facilities:
                    direct_cost = demand * self._get_distance_ij(i, j)
                    client_cost = min(client_cost, direct_cost)
                
                # Opción 2: Ir a puntos de recogida disponibles
                available_pickups = [k for k in pickups if k in self.algorithm.K_i[i]]
                
                for k in available_pickups:
                    if facilities:  # Solo si hay facilidades para servir el punto de recogida
                        # Encontrar la facilidad más cercana al punto de recogida
                        best_facility_cost = min(self.algorithm.d_kj[k][j] for j in facilities 
                                            if self.algorithm.d_kj[k][j] != float('inf'))
                        if best_facility_cost != float('inf'):
                            pickup_cost = demand * best_facility_cost
                            client_cost = min(client_cost, pickup_cost)
                
                # Si no hay opción válida para este cliente, penalizar fuertemente
                if client_cost == float('inf'):
                    total_cost = float('inf')
                    break
                
                total_cost += client_cost
            
            return total_cost
            
        except Exception as e:
            self.algorithm.logger.error(f"Error evaluando solución parcial: {e}")
            return float('inf')
    
    def evaluate(self):
        try:
            self.algorithm.logger.debug("Asignando clientes...")
            self._assign_customers()
            self.algorithm.logger.debug("Calculando costos...")
            
            if self.problem == "P1":
                cost = self._calculate_cost_P1()
            else:
                cost = self._calculate_cost_P2()
            
            self.cost = cost
            self.algorithm.logger.debug(f"Costo total calculado: {self.cost}")
            return cost
        
        except Exception as e:
            self.algorithm.logger.error(f"Error en evaluate: {e}")
            self.algorithm.logger.error(traceback.format_exc()) # Añade esta línea
            self.cost = float('inf')
            return self.cost
    
    def _assign_customers(self):
        """Asignación de clientes optimizada según el modelo P1"""
        
        self.algorithm.logger.debug(f"Instalaciones abiertas: {self.open_facilities}")
        self.algorithm.logger.debug(f"Puntos de recogida abiertos: {self.open_pickups}")
        
        for i in self.algorithm.I:
            try:
                # Limpieza optimizada de asignaciones previas
                self._clear_client_assignments(i)
                demand = self.algorithm.h[i-1]
                
                self.algorithm.logger.debug(f"Reasignando cliente {i}...")
                
                # Encontrar la mejor opción: asignación directa vs punto de recogida
                best_cost = float('inf')
                best_assignment = None
                
                # Opción 1: Asignación directa a una facilidad
                for j in self.open_facilities:
                    cost = self._get_distance_ij(i, j)
                    if cost < best_cost:
                        best_cost = cost
                        best_assignment = ('direct', j)
                
                # Opción 2: Ir a un punto de recogida accesible
                available_pickups = [k for k in self.open_pickups if k in self.algorithm.K_i[i]]
                
                for k in available_pickups:
                    # Encontrar la mejor facilidad para servir este punto de recogida
                    best_facility_for_k = min(self.open_facilities, key=lambda j: self.algorithm.d_kj[k][j])
                    cost = self.algorithm.d_kj[k][best_facility_for_k]
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_assignment = ('pickup', k, best_facility_for_k)
                
                # Ejecutar la mejor asignación
                if best_assignment[0] == 'direct':
                    j = best_assignment[1]
                    if self.algorithm.problem == "P1":
                        self.x[(i, j)] = 1
                    else:
                        self.w[(i, i, j)] = 1
                    self.algorithm.logger.debug(f"Cliente {i} asignado directamente a facilidad {j}")
                else:  # pickup
                    k, j = best_assignment[1], best_assignment[2]
                    if self.algorithm.problem == "P1":
                        self.z[(i, k)] = 1
                        self.s[(k, j)] += demand
                    else:
                        self.w[(i, k, j)] = 1
                    self.algorithm.logger.debug(f"Cliente {i} asignado a punto {k}, servido por facilidad {j}")
            except Exception as e:
                self.algorithm.logger.error(f"Error al asignar cliente {i}: {e}")
                self.algorithm.logger.error(traceback.format_exc())
                raise
    
    def _clear_client_assignments(self, i):
        """Limpia las asignaciones previas de un cliente"""
        try:
            if self.problem == "P1":
                # Limpiar x del cliente i
                for j in self.algorithm.J:
                    self.x[(i, j)] = 0
                
                # Limpiar z del cliente i y actualizar s correspondiente
                for k in self.algorithm.K_i[i]:
                    if (i, k) in self.z and self.z[(i, k)] == 1:
                        demand = self.algorithm.h[i-1]
                        # Encontrar qué facilidad servía este punto y reducir el flujo
                        for j in self.algorithm.J:
                            if self.s[(k, j)] >= demand:
                                self.s[(k, j)] -= demand
                                break
                    self.z[(i, k)] = 0
            else:  # P2
                # Limpiar w del cliente i
                keys_to_clear = [key for key in self.w.keys() if key[0] == i]
                for key in keys_to_clear:
                    self.w[key] = 0
        except Exception as e:
            self.algorithm.logger.error(f"Error al limpiar asignaciones del cliente {i}: {e}")
            self.algorithm.logger.error(traceback.format_exc())
            raise
    
    def _calculate_cost_P1(self):
        """Cálculo del costo según la formulación P1 del paper"""
        try:
            cost = 0
            
            # Primer término: Σ(i∈I) Σ(j∈J) h_i * d_ij * x_ij
            # Coste por asignaciones directas cliente → instalación
            for (i, j), value in self.x.items():
                if value == 1:
                    cost += self.algorithm.h[i-1] * self._get_distance_ij(i, j)
                    self.algorithm.logger.debug(f"Cliente {i} → Facilidad {j}: {self.algorithm.h[i-1]} * {self._get_distance_ij(i, j)} = {self.algorithm.h[i-1] * self._get_distance_ij(i, j)}")
            
            # Segundo término: Σ(k∈K) Σ(j∈J) d_kj * s_kj
            # Coste por transporte desde facilidades a puntos de recogida
            for (k, j), flow in self.s.items():
                if flow > 0:
                    transport_cost = self.algorithm.d_kj[k][j] * flow
                    cost += transport_cost
                    self.algorithm.logger.debug(f"Punto {k} ← Facilidad {j}: {flow} * {self.algorithm.d_kj[k][j]} = {transport_cost}")
            
            return cost
        except Exception as e:
            self.algorithm.logger.error(f"Error en cálculo de costo P1: {e}")
            self.algorithm.logger.error(traceback.format_exc())
            raise
    
    def _calculate_cost_P2(self):
        try:
            cost = 0
            for (i, k, j), value in self.w.items():
                if value == 1:
                    if k == i:
                        cost += self.algorithm.h[i-1] * self._get_distance_ij(i, j)
                    else:
                        cost += self.algorithm.h[i-1] * self.algorithm.d_kj[k][j]
            return cost
        except Exception as e:
            self.algorithm.logger.error(f"Error en cálculo de costo P2: {e}")
            self.algorithm.logger.error(traceback.format_exc())
            raise


def createEmptySolution(instance):
    solution = {}
    solution['sol'] = set()
    solution['of'] = 0 # representa el valor de la función objetivo (objective function) asociado con la solución
    solution['instance'] = instance
    return solution


def addToSolution(sol, u, ofVariation = -1):
    if ofVariation == -1:
        for s in sol['sol']:
            sol['of'] += sol['instance']['d'][u][s]
    else:
        sol['of'] += ofVariation
    sol['sol'].add(u)

def distanceToSolution(sol, u, without = -1):
    d = 0
    for s in sol['sol']:
        if s != without:
            print(f"s: {s}\n u: {u}")
            d += sol['instance']['Demand'][s][u+1]
    return round(d, 2)

def isFeasible(sol):

    return len(sol['sol']) == sol['instance']['p']


def printSol(sol):
    print("SOL: "+str(sol['sol']))
    print("OF: "+str(round(sol['of'],2)))