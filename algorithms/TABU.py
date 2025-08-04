import logging
from math import ceil
import os
import random
from collections import deque
from copy import deepcopy
import time
import numpy as np
from sklearn.cluster import KMeans

class Solution:
    def __init__(self, tabu, other_solution=None):
        self.tabu = tabu
        self.problem = tabu.problem
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
    
    def _initialize_empty(self):
        """Inicialización optimizada según el tipo de problema"""
        self.y = {j: 0 for j in self.tabu.J}
        self.nu = {k: 0 for k in self.tabu.K}
        
        if self.problem == "P1":
            self.x = {(i, j): 0 for i in self.tabu.I for j in self.tabu.J}
            self.z = {(k, j): 0 for k in self.tabu.K for j in self.tabu.J}
            self.s = {(k, j): 0 for k in self.tabu.K for j in self.tabu.J}
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
            return self.tabu.d_ij[i][j]
        else:
            return self.tabu.d_ij[j][i]
        
    def initialize_greedy(self, alpha=0.3):
        """Construcción greedy aleatorizada optimizada"""
        # Reset de variables de decisión
        for j in self.tabu.J:
            self.y[j] = 0
        for k in self.tabu.K:
            self.nu[k] = 0

        # --- Selección de instalaciones ---
        len_I = len(self.tabu.I)
        facility_scores = [
            (j, sum(self._get_distance_ij(i, j) for i in self.tabu.I) / len_I)
            for j in self.tabu.J
        ]
        facility_scores.sort(key=lambda x: x[1])

        rcl_size = max(1, int(alpha * len(facility_scores)))
        selected_facilities = set()

        for _ in range(self.tabu.p):
            # Construcción del RCL directamente filtrando ya los seleccionados
            rcl = [pair for pair in facility_scores if pair[0] not in selected_facilities][:rcl_size]
            if not rcl: break  # Seguridad por si faltan elementos
            j, _ = random.choice(rcl)
            self.y[j] = 1
            selected_facilities.add(j)

        self.tabu.logger.info(f"[TABU-Greedy] Instalaciones seleccionadas: {[j for j in selected_facilities]}")

        # --- Selección de puntos de recogida ---
        open_facilities = self.open_facilities
        if open_facilities:
            open_set = set(open_facilities)
            pickup_scores = []
            for k in self.tabu.K:
                dists = [self.tabu.d_kj[k][j] for j in open_set if self.tabu.d_kj[k][j] != float('inf')]
                avg_dist = sum(dists) / len(dists) if dists else float('inf')
                pickup_scores.append((k, avg_dist))
        else:
            pickup_scores = [(k, float('inf')) for k in self.tabu.K]

        pickup_scores.sort(key=lambda x: x[1])
        rcl_size_pickup = max(1, int(alpha * len(pickup_scores)))
        selected_pickups = set()

        for _ in range(self.tabu.t):
            rcl = [pair for pair in pickup_scores if pair[0] not in selected_pickups][:rcl_size_pickup]
            if not rcl: break
            k, _ = random.choice(rcl)
            self.nu[k] = 1
            selected_pickups.add(k)

        self.tabu.logger.info(f"[TABU-Greedy] Puntos de recogida seleccionados: {[k for k in selected_pickups]}")

        self.evaluate()

    def initialize_kmeans(self):
        """Inicialización usando K-means para una mejor distribución espacial"""
        try:
            # Coordenadas para instalaciones
            facility_coords = np.array([
                [j, sum(self._get_distance_ij(i, j) for i in self.tabu.I) / len(self.tabu.I)]
                for j in self.tabu.J
            ])
            facility_indices = np.array(list(self.tabu.J))

            if len(facility_coords) > self.tabu.p:
                kmeans_facilities = KMeans(n_clusters=self.tabu.p, random_state=42, n_init=10)
                kmeans_facilities.fit(facility_coords)
                centers = kmeans_facilities.cluster_centers_

                # Buscar la instalación más cercana a cada centroide
                selected_facilities = set()
                coords_matrix = facility_coords[:, None, :]  # shape: (n, 1, 2)
                centers_matrix = centers[None, :, :]          # shape: (1, k, 2)
                distances = np.linalg.norm(coords_matrix - centers_matrix, axis=2)  # shape: (n, k)

                closest_indices = np.argmin(distances, axis=0)
                selected_facilities.update(facility_indices[closest_indices])

                # Completar si faltan instalaciones
                if len(selected_facilities) < self.tabu.p:
                    remaining = [j for j in self.tabu.J if j not in selected_facilities]
                    avg_dist = {
                        j: sum(self._get_distance_ij(i, j) for i in self.tabu.I) / len(self.tabu.I)
                        for j in remaining
                    }
                    sorted_remaining = sorted(avg_dist, key=avg_dist.get)
                    selected_facilities.update(sorted_remaining[:self.tabu.p - len(selected_facilities)])
            else:
                selected_facilities = set(self.tabu.J)

            # Activar instalaciones seleccionadas
            for j in selected_facilities:
                self.y[j] = 1

            self.tabu.logger.info(f"[K-means] Instalaciones seleccionadas: {[j for j in selected_facilities]}")

            # Coordenadas para puntos de recogida
            if self.open_facilities:
                open_set = set(self.open_facilities)
                pickup_coords = np.array([
                    [k, sum(self.tabu.d_kj[k][j] for j in open_set if self.tabu.d_kj[k][j] != float('inf')) / len(open_set)]
                    for k in self.tabu.K
                ])
            else:
                pickup_coords = np.array([[k, 0] for k in self.tabu.K])

            pickup_indices = np.array(list(self.tabu.K))

            if len(pickup_coords) > self.tabu.t:
                kmeans_pickups = KMeans(n_clusters=self.tabu.t, random_state=42, n_init=10)
                kmeans_pickups.fit(pickup_coords)
                centers = kmeans_pickups.cluster_centers_

                coords_matrix = pickup_coords[:, None, :]
                centers_matrix = centers[None, :, :]
                distances = np.linalg.norm(coords_matrix - centers_matrix, axis=2)

                closest_indices = np.argmin(distances, axis=0)
                selected_pickups = set(pickup_indices[closest_indices])

                if len(selected_pickups) < self.tabu.t:
                    remaining = [k for k in self.tabu.K if k not in selected_pickups]
                    pickup_scores = {}
                    for k in remaining:
                        if self.open_facilities:
                            dists = [self.tabu.d_kj[k][j] for j in self.open_facilities if self.tabu.d_kj[k][j] != float('inf')]
                            pickup_scores[k] = sum(dists) / len(dists) if dists else float('inf')
                        else:
                            pickup_scores[k] = float('inf')
                    sorted_remaining = sorted(pickup_scores, key=pickup_scores.get)
                    selected_pickups.update(sorted_remaining[:self.tabu.t - len(selected_pickups)])
            else:
                selected_pickups = set(self.tabu.K)

            # Activar puntos de recogida seleccionados
            for k in selected_pickups:
                self.nu[k] = 1

            self.tabu.logger.info(f"[K-means] Puntos de recogida seleccionados: {[k for k in selected_pickups]}")

            self.evaluate()

        except Exception as e:
            self.tabu.logger.error(f"Error en inicialización K-means: {e}")
            self.initialize_greedy()

    def initialize_random(self):
        for j in random.sample(self.tabu.J, self.tabu.p):
            self.y[j] = 1
        self.tabu.logger.info(f"Instalaciones seleccionadas: {[j for j in self.tabu.J if self.y[j] == 1]}")

        for k in random.sample(self.tabu.K, self.tabu.t):
            self.nu[k] = 1
        self.tabu.logger.info(f"Puntos de recogida seleccionados: {[k for k in self.tabu.K if self.nu[k] == 1]}")

        self.evaluate()


    def evaluate(self, affected_clients=None):
        try:
            self.tabu.logger.debug("Asignando clientes...")
            self._assign_customers(affected_clients=affected_clients)

            self.tabu.logger.debug("Calculando costos...")
            if self.problem == "P1":
                cost = self._calculate_cost_P1()
            else:
                cost = self._calculate_cost_P2()

            self.cost = cost
            self.tabu.logger.info(f"Costo total calculado: {self.cost}")
            return cost
        
        except Exception as e:
            self.tabu.logger.error(f"Error en evaluate: {e}")
            self.cost = float('inf')
            return self.cost
        
    def _assign_customers(self, affected_clients=None):
        """Asignación de clientes optimizada"""
        if affected_clients is None:
            affected_clients = self.tabu.I
        
        self.tabu.logger.debug(f"Instalaciones abiertas: {self.open_facilities}")
        self.tabu.logger.debug(f"Puntos de recogida abiertos: {self.open_pickups}")
        
        for i in affected_clients:
            # Limpieza optimizada de asignaciones previas
            self._clear_client_assignments(i)
            
            self.tabu.logger.debug(f" Reasignando cliente {i}...")
            
            best_assignment = None
            best_facility = None
            best_pickup = None
            best_cost = float('inf')
            h_i = self.tabu.h[i - 1]
            
            # Caso 1: Asignación directa (i, i, j) o (i, j)
            for j in self.open_facilities:
                self.tabu.logger.debug(f" Evaluando instalación {j}...")
                dist = self._get_distance_ij(i, j)
                cost = h_i * dist
                
                if cost < best_cost:
                    best_assignment = (i, j) if self.problem == "P1" else (i, i, j)
                    best_cost = cost
                    best_facility = j
                    self.tabu.logger.debug(f"  Costo directo a instalación {j}: {cost}")
                    
                    if cost == 0:
                        break
            
            self.tabu.logger.debug(f"Mejor asignación directa para cliente {i}: {best_assignment} con costo {best_cost}")
            
            # Caso 2: Asignación vía punto de recogida
            for k in self.open_pickups:
                if k not in self.tabu.K_i[i]:
                    continue
                
                self.tabu.logger.debug(f" Evaluando punto de recogida {k}...")
                
                try:
                    dist_ik = self.tabu.d_kj[k][i]
                except AttributeError:
                    self.tabu.logger.error(f"Error: self.tabu.d_ik no encontrado. No se puede calcular correctamente el costo para asignaciones a puntos de recogida para el cliente {i} y punto de recogida {k}.")

                for j in self.open_facilities:
                    dist_kj = self.tabu.d_kj[k][j]
                    cost = h_i * (dist_ik + dist_kj)
                    self.tabu.logger.debug(f"  Costo a punto de recogida {k}: {cost}")
                    
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
        for j in self.tabu.J:
            if self.y[j] == 1:
                dist = self.tabu.d_kj[k][j]
                cost = self.tabu.h[k - 1] * dist
                if cost < best_cost:
                    best_cost = cost
                    best_facility = j
        return best_facility
    
    def _update_assignment(self, i, best_assignment, best_pickup, best_facility, h_i):
        """Actualiza las estructuras de datos con la mejor asignación"""
        self.tabu.logger.debug(f"Asignando {best_assignment}...")
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
        
        for j in self.tabu.J:
            if self.y[j] == 1:
                dist = self.tabu.d_kj[k][j]
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
                    cost += self.tabu.h[i-1] * self._get_distance_ij(i, j)
                else:
                    cost += self.tabu.h[i-1] * self.tabu.d_kj[k][j]
        return cost

    def _calculate_cost_P1(self):
        cost = 0
        for (i, j), value in self.x.items():
            if value == 1:
                cost += self.tabu.h[i-1] * self._get_distance_ij(i, j)
        for (k, j), value in self.s.items():
            cost += self.tabu.d_kj[k][j] * value
        return cost
    
class TabuSearch:
    def __init__(self, params, inicialization, instance, result_dir, logger: logging, problem: str = "P2", max_iter=None, tabu_tenure=7):
        self.I = params['I']
        self.J = params['J']
        self.K = params['K']
        self.h = params['h']
        self.d_ij = params['d_ij']
        self.d_kj = params['d_kj']
        self.K_i = params['K_i']
        self.p = params['p']
        self.t = params['t']

        self.max_iter = max_iter if max_iter else 2*len(self.I)
        self.x = 10 if len(self.I) < 30 else 15
        self.tabu_tenure = tabu_tenure
        self.tabu_list = deque(maxlen=tabu_tenure)

        self.best_solution = None
        self.current_solution = None
        self.history = []
        self.inicialization = inicialization

        self.result_dir = result_dir
        self.instance = instance
        self.problem = problem
        self.logger = logger

    def run(self):
        start_time = time.time()
        self.current_solution = Solution(self)

        # --- Inicialización de la solución ---
        if self.inicialization == "greedy":
            self.current_solution.initialize_greedy()
        elif self.inicialization == "random":
            self.current_solution.initialize_random()
        elif self.inicialization == "kmeans":
            self.current_solution.initialize_kmeans()
        else:
            raise ValueError(f"Inicialización desconocida: {self.inicialization}")
        
        self.best_solution = Solution(self, other_solution=self.current_solution)
        stagnation_counter = 0

        # --- Bucle principal de búsqueda ---
        for iteration in range(1, self.max_iter):
            self.logger.info(f"Iteración {iteration + 1}/{self.max_iter}, Costo actual: {self.current_solution.cost}, Mejor costo: {self.best_solution.cost}")

            neighborhood, _ = self._generate_neighborhood()
            self.logger.info(f"Vecindario generado con {len(neighborhood)} candidatos")

            best_candidate = None
            best_candidate_cost = float('inf')
            best_move = None

            for candidate, move in neighborhood:
                candidate_cost = candidate.evaluate()
                is_tabu = self._is_tabu(move)
                aspiration = candidate_cost < self.best_solution.cost

                if (not is_tabu) or aspiration:
                    if candidate_cost < best_candidate_cost:
                        best_candidate = candidate
                        best_candidate_cost = candidate_cost
                        best_move = move

            if best_candidate is not None:
                self.current_solution = best_candidate

                if best_candidate_cost < self.best_solution.cost:
                    #CAMBIAR
                    self.best_solution = deepcopy(best_candidate)
                    self.logger.info(f"Nueva mejor solución encontrada: {best_candidate_cost}")
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                self.tabu_list.append(best_move)

            self.history.append(self.best_solution.cost)

            # --- Estancamiento: aplicar perturbación ---
            if stagnation_counter >= self.x:
                self.logger.info(f"Estancamiento detectado en iteración {iteration}, aplicando perturbación aleatoria")
                perturbed_solution = self._perturb_randomly()
                self.current_solution = perturbed_solution
                stagnation_counter = 0

            # --- Criterio de parada por falta de mejora ---    
            if iteration > 20 and len(set(self.history[-20:])) == 1:
                break

        solve_time = time.time() - start_time

        if self.best_solution is None:
            self.logger.warning("Error: No se encontró ninguna solución válida después de todas las iteraciones.")
            return None
        
        self._save_best_sol(solve_time)

    def _generate_neighborhood(self):
        neighborhood = []
        affected_clients = set()
        
        num_changes_y = ceil(len(self.current_solution.open_facilities) / 2)
        num_changes_nu = ceil(len(self.current_solution.open_pickups) / 2)
        
        # --- Swap de instalaciones (cerrar la peor, abrir una buena) ---
        if self.current_solution.open_facilities and self.current_solution.closed_facilities:
            def media_asignacion_instalacion(j):
                if self.problem == "P2":
                    clientes = [i for i in self.I if self.current_solution.w.get((i, i, j), 0) == 1]
                elif self.problem == "P1":
                    clientes = [i for i in self.I if self.current_solution.x.get((i, j), 0) == 1]
                else:
                    clientes = [i for i in self.I if self.current_solution.w.get((i, i, j), 0) == 1]
                if not clientes:
                    return 0
                return sum(self.current_solution._get_distance_ij(i, j) for i in clientes) / len(clientes)
            
            j_peor = max(self.current_solution.open_facilities, key=media_asignacion_instalacion)
            # j_peor = list(self.current_solution.open_facilities)[0]
            
            # Seleccionamos las instalaciones cerradas más prometedoras
            def score_instalacion(j):
                return sum(
                    self.current_solution._get_distance_ij(i, j) for i in self.I
                ) / len(self.I)
            
            top_candidates = sorted(self.current_solution.closed_facilities, key=score_instalacion)[:10]
            candidatas = random.sample(top_candidates, min(num_changes_y, len(top_candidates)))

            for j_nueva in candidatas:
                new_sol = Solution(self, other_solution=self.current_solution)
                new_sol.y[j_peor] = 0
                new_sol.y[j_nueva] = 1
                
                if self.problem == "P1":
                    affected_clients.update(
                        i for i in self.I if self.current_solution.x.get((i, j_peor), 0) == 1
                    )
                else:
                    affected_clients.update(
                        i for i in self.I if self.current_solution.w.get((i, i, j_peor), 0) == 1
                    )
                neighborhood.append((new_sol, ('instalacion_heuristica', j_peor, j_nueva)))

        # --- Parte heurística: swap guiado para puntos de recogida ---
        if self.current_solution.open_pickups and self.current_solution.closed_pickups:
            def media_asignacion_punto(k):
                clientes = []
                for i in self.I:
                    if self.problem == "P1" and self.current_solution.z.get((i, k), 0) == 1 and k in self.K_i[i]:
                        clientes.append(i)
                    elif self.problem == "P2":
                        for j in self.J:
                            if self.current_solution.w.get((i, k, j), 0) == 1:
                                clientes.append(i)
                                break
                if not clientes:
                    return 0
                return sum(self.d_kj[k][i] for i in clientes) / len(clientes)
            
            puntos_abiertos_actuales = list(self.current_solution.open_pickups)
            k_peor = max(puntos_abiertos_actuales, key=media_asignacion_punto)
            
            # puntos_abiertos_actuales = [k for k in self.current_solution.open_pickups 
            #                         if self.current_solution.nu.get(k, 0) == 1]
            
            # if puntos_abiertos_actuales:
            #     k_peor = max(puntos_abiertos_actuales, key=media_asignacion_punto)
            # else:
            #     # Si no hay puntos abiertos consistentes, tomar el primero disponible
            #     k_peor = list(self.current_solution.open_pickups)[0]
            
            def score_punto(k):
                distancias = [self.d_kj[k][i] for i in self.I if k in self.d_kj and i in self.d_kj[k]]
                return sum(distancias) / len(distancias) if distancias else float('inf')
            
            top_candidates = sorted(self.current_solution.closed_pickups, key=score_punto)[:10]
            candidatas = random.sample(top_candidates, min(num_changes_nu, len(top_candidates)))
            
            for k_nuevo in candidatas:
                new_sol = Solution(self, other_solution=self.current_solution)
                new_sol.nu[k_peor] = 0
                new_sol.nu[k_nuevo] = 1
                
                for i in self.I:
                    if self.problem == "P1" and self.current_solution.z.get((i, k_peor), 0) == 1 and k_peor in self.K_i[i]:
                        affected_clients.add(i)
                    elif self.problem == "P2":
                        for j in self.J:
                            if self.current_solution.w.get((i, k_peor, j), 0) == 1:
                                affected_clients.add(i)
                neighborhood.append((new_sol, ('punto_heuristica', k_peor, k_nuevo)))

        return neighborhood, list(affected_clients)

    def _perturb_randomly(self):
        new_sol = Solution(self, other_solution=self.current_solution)

        num_changes_y = ceil(len(self.current_solution.open_facilities) / 2)
        num_changes_nu = ceil(len(self.current_solution.open_pickups) / 2)

        cerrar_y = random.sample(self.current_solution.open_facilities, min(num_changes_y, len(self.current_solution.open_facilities)))
        abrir_y = random.sample(self.current_solution.closed_facilities, min(num_changes_y, len(self.current_solution.closed_facilities)))
        cerrar_nu = random.sample(self.current_solution.open_pickups, min(num_changes_nu, len(self.current_solution.open_pickups)))
        abrir_nu = random.sample(self.current_solution.closed_pickups, min(num_changes_nu, len(self.current_solution.closed_pickups)))

        for j in cerrar_y:
            new_sol.y[j] = 0
        for j in abrir_y:
            new_sol.y[j] = 1
        for k in cerrar_nu:
            new_sol.nu[k] = 0
        for k in abrir_nu:
            new_sol.nu[k] = 1
        new_sol.evaluate()

        # Verificación defensiva
        # print("Instalaciones abiertas:", new_sol.open_facilities)
        # print("Cantidad de instalaciones que deberían estar abiertas:", self.p)
        # print("Puntos de recogida abiertos:", new_sol.open_pickups)
        # print("Cantidad de puntos de recogida que deberían estar abiertos:", self.t)
        assert sum(new_sol.y[j] for j in self.J) == self.p, "¡El número de instalaciones abiertas no es p!"
        assert sum(new_sol.nu[k] for k in self.K) == self.t, "¡El número de puntos de recogida abiertos no es t!"
        return new_sol
    
    def _is_tabu(self, move):
        return move in self.tabu_list
    
    def _save_best_sol(self, solve_time):
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        filename = f"{self.instance+1}_best_solution.txt"
        filepath = os.path.join(self.result_dir, filename)
        with open(filepath, 'w') as f:
            f.write(f"Instancia: {self.instance+1}\n")
            f.write(f"Costo mejor solución: {self.best_solution.cost}\n")
            f.write(f"Tiempo ejecución (seg): {solve_time:.2f}\n")
            f.write("Instalaciones abiertas:\n")
            for j, val in self.best_solution.y.items():
                if val == 1:
                    f.write(f"  Instalación {j}\n")
            f.write("Puntos de recogida abiertos:\n")
            for k, val in self.best_solution.nu.items():
                if val == 1:
                    f.write(f"  Punto {k}\n")

# Ejemplo uso (debes adaptar los parámetros a tu instancia)
# params = {
#     'I': list(range(num_clientes)),
#     'J': list(range(num_instalaciones)),
#     'K': list(range(num_puntos)),
#     'h': demandas_clientes,
#     'd_ij': matriz_dist_cliente_instalacion,
#     'd_kj': matriz_dist_punto_instalacion,
#     'K_i': puntos_por_cliente,
#     'I_k': clientes_por_punto,
#     'p': num_instalaciones_a_abrir,
#     't': num_puntos_a_abrir
# }
# ts = TabuSearch(params, instance="Instancia1", result_dir="./resultados", max_iter=100)
# ts.run()
# print("Mejor costo encontrado:", ts.best_solution.cost)

