import random
import time
import os

class Solution:
    def __init__(self, grasp, other_solution=None):
        self.grasp = grasp
        self.problem = grasp.problem
        self.cost = float('inf')
        self.time = 0
        self.client_assignments = {}

        if other_solution is None:
            self._initialize_empty()
        else:
            self._clone_solution(other_solution)

    @property
    def open_facilities(self):
        return [j for j, val in self.y.items() if val == 1]

    @property
    def closed_facilities(self):
        return [j for j, val in self.y.items() if val == 0]
    
    @property
    def open_pickups(self):
        return [k for k, val in self.nu.items() if val == 1]

    @property
    def closed_pickups(self):
        return [k for k, val in self.nu.items() if val == 0]
    
    def _set_nu(self, k, value):
        """Activa o desactiva el punto de recogida k y actualiza las asignaciones afectadas"""
        if self.nu[k] == value:
            return  # No hay cambio, evitamos trabajo innecesario

        self.nu[k] = value

        # Identificar los clientes que pueden verse afectados
        affected_clients = [i for i in self.grasp.I if k in self.grasp.K_i[i]]

        # Reasignar solo los clientes afectados y recalcular coste
        self.evaluate(affected_clients=affected_clients)

    
    def _set_y(self, j, value):
        """Activa o desactiva la instalación j y actualiza las asignaciones afectadas"""
        if self.y[j] == value:
            return  # No hay cambio

        self.y[j] = value

        # Clientes afectados: todos, ya que todos pueden asignarse a cualquier j ∈ J abierta
        self.evaluate()

    def _initialize_empty(self):
        """Inicialización optimizada según el tipo de problema"""
        self.y = {j: 0 for j in self.grasp.J}
        self.nu = {k: 0 for k in self.grasp.K}
        
        if self.problem == "P1":
            self.x = {(i, j): 0 for i in self.grasp.I for j in self.grasp.J}
            self.z = {(k, j): 0 for k in self.grasp.K for j in self.grasp.J}
            self.s = {(k, j): 0 for k in self.grasp.K for j in self.grasp.J}
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
            return self.grasp.d_ij[i][j]
        else:
            return self.grasp.d_ij[j][i]
        
    def initialize_greedy(self, alpha=0.3):
        """Construcción greedy aleatorizada optimizada"""
        # Reset de variables de decisión
        for j in self.grasp.J:
            self.y[j] = 0
        for k in self.grasp.K:
            self.nu[k] = 0

        # --- Selección de instalaciones ---
        len_I = len(self.grasp.I)
        facility_scores = [
            (j, sum(self._get_distance_ij(i, j) for i in self.grasp.I) / len_I)
            for j in self.grasp.J
        ]
        facility_scores.sort(key=lambda x: x[1])

        rcl_size = max(1, int(alpha * len(facility_scores)))
        selected_facilities = set()

        for _ in range(self.grasp.p):
            # Construcción del RCL directamente filtrando ya los seleccionados
            rcl = [pair for pair in facility_scores if pair[0] not in selected_facilities][:rcl_size]
            if not rcl: break  # Seguridad por si faltan elementos
            j, _ = random.choice(rcl)
            self.y[j] = 1
            selected_facilities.add(j)

        self.grasp.logger.info(f"[GRASP] Instalaciones seleccionadas: {[j for j in selected_facilities]}")

        # --- Selección de puntos de recogida ---
        open_facilities = self.open_facilities
        if open_facilities:
            open_set = set(open_facilities)
            pickup_scores = []
            for k in self.grasp.K:
                dists = [self.grasp.d_kj[k][j] for j in open_set if self.grasp.d_kj[k][j] != float('inf')]
                avg_dist = sum(dists) / len(dists) if dists else float('inf')
                pickup_scores.append((k, avg_dist))
        else:
            pickup_scores = [(k, float('inf')) for k in self.grasp.K]

        pickup_scores.sort(key=lambda x: x[1])
        rcl_size_pickup = max(1, int(alpha * len(pickup_scores)))
        selected_pickups = set()

        for _ in range(self.grasp.t):
            rcl = [pair for pair in pickup_scores if pair[0] not in selected_pickups][:rcl_size_pickup]
            if not rcl: break
            k, _ = random.choice(rcl)
            self.nu[k] = 1
            selected_pickups.add(k)

        self.grasp.logger.info(f"[GRASP] Puntos de recogida seleccionados: {[k for k in selected_pickups]}")

        self.evaluate()

    def evaluate(self, affected_clients=None):
        try:
            self.grasp.logger.debug("Asignando clientes...")
            self._assign_customers(affected_clients=affected_clients)

            self.grasp.logger.debug("Calculando costos...")
            if self.problem == "P1":
                cost = self._calculate_cost_P1()
            else:
                cost = self._calculate_cost_P2()

            self.cost = cost
            self.grasp.logger.info(f"Costo total calculado: {self.cost}")
            return cost
        
        except Exception as e:
            self.grasp.logger.error(f"Error en evaluate: {e}")
            self.cost = float('inf')
            return self.cost
        
    def _assign_customers(self, affected_clients=None):
        """Asignación de clientes optimizada"""
        if affected_clients is None:
            affected_clients = self.grasp.I
        
        self.grasp.logger.debug(f"Instalaciones abiertas: {self.open_facilities}")
        self.grasp.logger.debug(f"Puntos de recogida abiertos: {self.open_pickups}")
        
        for i in affected_clients:
            # Limpieza optimizada de asignaciones previas
            self._clear_client_assignments(i)
            
            self.grasp.logger.debug(f" Reasignando cliente {i}...")
            
            best_assignment = None
            best_facility = None
            best_pickup = None
            best_cost = float('inf')
            h_i = self.grasp.h[i - 1]
            
            # Caso 1: Asignación directa (i, i, j) o (i, j)
            for j in self.open_facilities:
                self.grasp.logger.debug(f" Evaluando instalación {j}...")
                dist = self._get_distance_ij(i, j)
                cost = h_i * dist
                
                if cost < best_cost:
                    best_assignment = (i, j) if self.problem == "P1" else (i, i, j)
                    best_cost = cost
                    best_facility = j
                    self.grasp.logger.debug(f"  Costo directo a instalación {j}: {cost}")
                    
                    if cost == 0:
                        break
            
            self.grasp.logger.debug(f"Mejor asignación directa para cliente {i}: {best_assignment} con costo {best_cost}")
            
            # Caso 2: Asignación vía punto de recogida
            for k in self.open_pickups:
                if k not in self.grasp.K_i[i]:
                    continue
                
                self.grasp.logger.debug(f" Evaluando punto de recogida {k}...")
                
                try:
                    dist_ik = self.grasp.d_kj[k][i]
                except AttributeError:
                    self.grasp.logger.error(f"Error: self.grasp.d_ik no encontrado. No se puede calcular correctamente el costo para asignaciones a puntos de recogida para el cliente {i} y punto de recogida {k}.")

                for j in self.open_facilities:
                    dist_kj = self.grasp.d_kj[k][j]
                    cost = h_i * (dist_ik + dist_kj)
                    self.grasp.logger.debug(f"  Costo a punto de recogida {k}: {cost}")
                    
                    if cost < best_cost:
                        best_assignment = (i, k) if self.problem == "P1" else (i, k, j)
                        best_cost = cost
                        best_pickup = k
                        best_facility = j
            
            # Actualización optimizada de asignaciones
            if best_assignment:
                self._update_assignment(i, best_assignment, best_pickup, best_facility, h_i)

    def _clear_client_assignments(self, i):
        """Limpia las asignaciones previas de un cliente"""
        if self.problem == "P1":
            # Limpiar x e z del cliente i
            keys_to_clear = [key for key in self.x.keys() if key[0] == i]
            for key in keys_to_clear:
                self.x[key] = 0
            
            keys_to_clear = [key for key in self.z.keys() if key[0] == i]
            for key in keys_to_clear:
                self.z[key] = 0
        else:  # P2
            # Limpiar w del cliente i
            keys_to_clear = [key for key in self.w.keys() if key[0] == i]
            for key in keys_to_clear:
                self.w[key] = 0

    def _assign_best_facility(self, k):
        """Asigna el mejor facility para un punto de recogida k"""
        best_cost = float('inf')
        best_facility = None
        for j in self.grasp.J:
            if self.y[j] == 1:
                dist = self.grasp.d_kj[k][j]
                cost = self.grasp.h[k - 1] * dist
                if cost < best_cost:
                    best_cost = cost
                    best_facility = j
        return best_facility
    
    def _update_assignment(self, i, best_assignment, best_pickup, best_facility, h_i):
        """Actualiza las estructuras de datos con la mejor asignación"""
        self.grasp.logger.debug(f"Asignando {best_assignment}...")
        self.client_assignments[i] = best_assignment

        if self.problem == "P1":
            if best_pickup is not None:
                self.z[(best_assignment[1], best_assignment[0])] = 1
                # best_facility = self._assign_best_facility(best_pickup, h_i)
                self.s[(best_pickup, best_facility)] += h_i
            else:
                self.x[best_assignment] = 1
        else:  # P2
            self.w[best_assignment] = 1
    
    def _assign_best_facility(self, k, h_i):
        """Asigna la mejor instalación para un punto de recogida k"""
        best_cost = float('inf')
        best_facility = None
        
        for j in self.grasp.J:
            if self.y[j] == 1:
                dist = self.grasp.d_kj[k][j]
                cost = h_i * dist
                if cost < best_cost:
                    best_cost = cost
                    best_facility = j
        
        return best_facility
    
    def _calculate_cost_P2(self):
        cost = 0

        for (i, k, j), value in self.w.items():
            if value == 1:
                if k == i:
                    cost += self.grasp.h[i-1] * self._get_distance_ij(i, j)
                else:
                    cost += self.grasp.h[i-1] * self.grasp.d_kj[k][j]
        return cost

    def _calculate_cost_P1(self):
        cost = 0
        # Coste por asignaciones directas cliente -> instalación
        for (i, j), value in self.x.items():
            if value == 1:
                cost += self.grasp.h[i - 1] * self._get_distance_ij(i, j)

        # Coste por asignaciones cliente -> pickup -> instalación
        for (k, j), flow in self.s.items():
            if flow > 0:
                cost += self.grasp.d_kj[k][j] * flow

        return cost


class GRASPSearch:
    def __init__(self, params, instance, result_dir, logger, frac_neighbors: int = 4, 
                 alpha: float = 0.3, problem="P2", max_iter=None):
        # Asignación directa de parámetros
        self.I = params['I']
        self.J = params['J']
        self.K = params['K']
        self.h = params['h']
        self.d_ij = params['d_ij']
        self.d_kj = params['d_kj']
        self.K_i = params['K_i']
        self.p = params['p']
        self.t = params['t']
        
        # Configuración del algoritmo
        self.alpha = alpha
        self.frac_neighbors = frac_neighbors
        self.max_iter = max_iter if max_iter else len(self.I)
        self.num_neighbors = max(1, len(self.I) // self.frac_neighbors)
        
        # Estado del algoritmo
        self.best_solution = None
        self.history = []
        
        # Configuración de salida
        self.result_dir = result_dir
        self.instance = instance
        self.problem = problem
        self.logger = logger
        
        # Pre-computar límites para optimización
        self._max_facility_swaps = min(self.num_neighbors, len(self.J) * len(self.J))
        self._max_pickup_swaps = min(self.num_neighbors, len(self.K) * len(self.K))
    
    def run(self):
        """Ejecución principal del algoritmo GRASP"""
        start_time = time.time()
        self.best_solution = None
        
        for iteration in range(1, self.max_iter + 1):
            self.logger.info(f"Iteración {iteration}/{self.max_iter}")
            
            # Construcción y mejora de solución
            current_solution = Solution(self)
            current_solution.initialize_greedy(alpha=self.alpha)
            current_solution = self.local_search(current_solution)
            
            self.logger.info(f"Costo de la solución encontrada: {current_solution.cost}")
            
            # Actualización de la mejor solución
            if self.best_solution is None or current_solution.cost < self.best_solution.cost:
                self.logger.info(f"¡Nueva mejor solución encontrada en iteración {iteration} con costo {current_solution.cost}!")
                self.best_solution = Solution(self, other_solution=current_solution)
            
            elapsed = time.time() - start_time
            self.best_solution.time = elapsed
            self.logger.info(f"Tiempo transcurrido: {elapsed:.2f} segundos")
        
        self.logger.info(f"Mejor costo encontrado: {self.best_solution.cost}")
        self._save_best_sol(solve_time=elapsed)
    
    def local_search(self, solution):
        """Búsqueda local optimizada"""
        current = Solution(self, other_solution=solution)
        improved = True
        
        while improved:
            improved = False
            best_neighbor = None
            best_cost = current.cost
            
            # Generar y evaluar vecinos con límite dinámico
            neighbor_count = 0
            for neighbor in self.generate_neighbors(solution=current, num_neighbors=self.num_neighbors):
                neighbor.evaluate()
                neighbor_count += 1
                
                if neighbor.cost < best_cost:
                    best_cost = neighbor.cost
                    best_neighbor = neighbor
                
                # Límite dinámico basado en mejoras encontradas
                if neighbor_count >= self.num_neighbors and best_neighbor is not None:
                    break
            
            # Actualizar si se encontró mejora
            if best_neighbor:
                current = best_neighbor
                improved = True
        
        return current
    
    def generate_neighbors(self, solution, num_neighbors=20):
        """Generación optimizada de vecinos con evitar duplicados eficientemente"""
        neighbors = []
        attempts = 0
        max_attempts = num_neighbors * 3  # Límite para evitar loops infinitos
        
        # Usar conjuntos para tracking más eficiente
        seen_facility_configs = set()
        seen_pickup_configs = set()
        
        while len(neighbors) < num_neighbors and attempts < max_attempts:
            attempts += 1
            
            # Alternar entre tipos de vecinos para variedad
            neighbor_type = attempts % 2
            
            if neighbor_type == 0 and solution.open_facilities and solution.closed_facilities:
                # Intercambio de instalaciones
                num_to_swap = random.randint(1, min(self.p, len(solution.open_facilities), len(solution.closed_facilities)))
                
                facilities_to_close = random.sample(solution.open_facilities, num_to_swap)
                facilities_to_open = random.sample(solution.closed_facilities, num_to_swap)
                
                # Crear configuración como tupla ordenada
                config = tuple(sorted([(f, 0) for f in facilities_to_close] + 
                                    [(f, 1) for f in facilities_to_open]))
                
                if config not in seen_facility_configs:
                    seen_facility_configs.add(config)
                    
                    neighbor = Solution(self, other_solution=solution)
                    for facility in facilities_to_close:
                        neighbor._set_y(facility, 0)
                    for facility in facilities_to_open:
                        neighbor._set_y(facility, 1)
                    
                    neighbors.append(neighbor)
                    
            elif neighbor_type == 1 and solution.open_pickups and solution.closed_pickups:
                # Intercambio de puntos de recogida
                num_to_swap = random.randint(1, min(self.t, len(solution.open_pickups), len(solution.closed_pickups)))
                
                pickups_to_close = random.sample(solution.open_pickups, num_to_swap)
                pickups_to_open = random.sample(solution.closed_pickups, num_to_swap)
                
                # Crear configuración como tupla ordenada
                config = tuple(sorted([(p, 0) for p in pickups_to_close] + 
                                    [(p, 1) for p in pickups_to_open]))
                
                if config not in seen_pickup_configs:
                    seen_pickup_configs.add(config)
                    
                    neighbor = Solution(self, other_solution=solution)
                    for pickup in pickups_to_close:
                        neighbor._set_nu(pickup, 0)
                    for pickup in pickups_to_open:
                        neighbor._set_nu(pickup, 1)
                    
                    neighbors.append(neighbor)
        
        return neighbors
    
    def _save_best_sol(self, solve_time):
        """Guardado optimizado de la mejor solución"""
        os.makedirs(self.result_dir, exist_ok=True)
        
        filename = f"{self.instance+1}_best_solution.txt"
        filepath = os.path.join(self.result_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(f"Instancia: {self.instance+1}\n")
            f.write(f"Número de vecinos: {self.num_neighbors}\n")
            f.write(f"Alfa: {self.alpha}\n")
            f.write(f"Costo mejor solución: {self.best_solution.cost}\n")
            f.write(f"Tiempo ejecución (seg): {solve_time:.2f}\n")
            
            f.write("Instalaciones abiertas:\n")
            for j in self.best_solution.open_facilities:
                f.write(f"  Instalación {j}\n")
            
            f.write("Puntos de recogida abiertos:\n")
            for k in self.best_solution.open_pickups:
                f.write(f"  Punto {k}\n")