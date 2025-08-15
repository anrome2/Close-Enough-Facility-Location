import logging
from math import ceil
import os
import random
from collections import deque
import time

import numpy as np

from structure.solution import Solution
    
class TabuSearch:
    def __init__(self, params, inicialization, instance, result_dir, logger: logging, problem: str = "P2", max_iter=None, tabu_tenure=0.25):
        self.I = params['I']
        self.J = params['J']
        self.K = params['K']
        self.h = params['h']
        self.d_ij = params['d_ij']
        self.d_kj = params['d_kj']
        self.K_i = params['K_i']
        self.p = params['p']
        self.t = params['t']
        self.n_pickups = len(params['K'])
        self.n_customers = len(self.I)

        # Parámetros de configuración
        self.max_iter = max_iter if max_iter else max(100, 2*len(self.I))
        self.max_iter_without_improvement = 10 if len(self.I) < 30 else 15

        # Tabu tenure dinámico basado en el tamaño del problema
        base_tenure = max(5, int(tabu_tenure * len(self.I)))
        self.tabu_tenure_facilities = base_tenure
        self.tabu_tenure_pickups = max(3, int(base_tenure * 0.7)) 

        # Listas tabu con tenure diferenciado
        self.tabu_list_facilities = deque(maxlen=self.tabu_tenure_facilities)
        self.tabu_list_pickup = deque(maxlen=self.tabu_tenure_pickups)

        # Memoria a largo plazo
        self.RC_install = np.zeros(self.n_customers, dtype=int)
        self.RC_pickup = np.zeros(self.n_pickups, dtype=int)
        self.AO_install = {j: [] for j in self.J}
        self.AO_pickup = {k: [] for k in self.K}
        
        # Parámetros para diversificación (según el paper)
        self.gamma_f = 0.75  # Factor de frecuencia
        self.gamma_q = 0.5   # Factor de calidad
        
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
        
        # Inicialización
        self.current_solution = Solution(self)
        self._initialize_solution()

        self.best_solution = Solution(self, other_solution=self.current_solution)
        stagnation_counter = 0

        self.logger.info(f"Iniciando Tabu Search - Solución inicial: {self.current_solution.cost}")

        # --- Bucle principal de búsqueda ---
        for iteration in range(1, self.max_iter+1):
            self.logger.info(f"Iteración {iteration}/{self.max_iter}, Costo actual: {self.current_solution.cost}, Mejor costo: {self.best_solution.cost}")

            # Fase de corto plazo (intensificación)
            self._short_term_phase()

            # Actualizar mejor solución
            if self.current_solution.cost < self.best_solution.cost:
                self.best_solution._clone_solution(self.current_solution)
                self.logger.info(f"Nueva mejor solución encontrada: {self.current_solution.cost}")
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Actualizar memoria a largo plazo
            self._long_term_memory()
            
            # Diversificación si hay estancamiento
            if stagnation_counter >= self.max_iter_without_improvement:
                self.logger.info(f"Estancamiento detectado, aplicando diversificación")
                self._long_term_phase()
                stagnation_counter = 0

        solve_time = time.time() - start_time

        if self.best_solution is None:
            self.logger.warning("No se encontró solución válida")
            return None
        
        self._save_best_sol(solve_time)

    def _initialize_solution(self):
        """Inicialización con manejo de errores mejorado"""
        try:
            if self.inicialization == "greedy":
                self.current_solution.initialize_greedy()
            elif self.inicialization == "random":
                self.current_solution.initialize_random()
            elif self.inicialization == "kmeans":
                self.current_solution.initialize_kmeans()
            else:
                self.logger.warning(f"Inicialización {self.inicialization} desconocida, usando greedy")
                self.current_solution.initialize_greedy()
        except Exception as e:
            self.logger.error(f"Error en inicialización {self.inicialization}: {e}")
            self.logger.info("Fallback a inicialización greedy")
            self.current_solution.initialize_greedy()

    def _short_term_phase(self):
        """Fase de intensificación mejorada"""
        # Swap de instalaciones
        self._perform_facility_swap()
        
        # Swap de puntos de recogida
        self._perform_pickup_swap()

    def _perform_facility_swap(self):
        """Realiza intercambio de instalaciones optimizado"""
        if not self.current_solution.open_facilities or not self.current_solution.closed_facilities:
            return
        
        # Obtener candidatos a cerrar (peores instalaciones)
        candidates_to_close = self._get_facility_candidates_to_close()
        if not candidates_to_close:
            return
        
        # Encontrar mejor instalación no-tabú para abrir
        best_facility_to_open = self._get_best_non_tabu_facility()
        if best_facility_to_open is None:
            return
        
        # Evaluar todos los swaps posibles y elegir el mejor
        best_swap_cost = float('inf')
        best_facility_to_close = None
        best_solution = Solution(self, other_solution=self.current_solution)
        
        for facility_to_close in candidates_to_close:
            # Crear solución temporal
            temp_solution = Solution(self, other_solution=self.current_solution)
            temp_solution.y[facility_to_close - 1] = 0
            temp_solution.y[best_facility_to_open - 1] = 1
            
            cost = temp_solution.evaluate()
            
            if cost < best_swap_cost:
                best_swap_cost = cost
                best_facility_to_close = facility_to_close
                best_solution._clone_solution(other=temp_solution)
        
        # Aplicar el mejor swap
        if best_facility_to_close is not None:
            self.current_solution._clone_solution(best_solution)
            self._add_tabu(best_facility_to_close, "facility")
            self.logger.debug(f"Swap instalación: cerrada {best_facility_to_close}, abierta {best_facility_to_open}")

    def _perform_pickup_swap(self):
        """Realiza intercambio de puntos de recogida optimizado"""
        if not self.current_solution.open_pickups or not self.current_solution.closed_pickups:
            return
        
        # Obtener candidatos a cerrar
        candidates_to_close = self._get_pickup_candidates_to_close()
        if not candidates_to_close:
            return
        
        # Encontrar mejor punto de recogida no-tabú para abrir
        best_pickup_to_open = self._get_best_non_tabu_pickup()
        if best_pickup_to_open is None:
            return
        
        # Evaluar swaps y elegir el mejor
        best_swap_cost = float('inf')
        best_pickup_to_close = None
        best_solution = None
        
        for pickup_to_close in candidates_to_close:
            # Crear solución temporal
            temp_solution = Solution(self, other_solution=self.current_solution)
            
            idx_close = self.K.index(pickup_to_close)
            idx_open = self.K.index(best_pickup_to_open)
            
            temp_solution.nu[idx_close] = 0
            temp_solution.nu[idx_open] = 1
            
            cost = temp_solution.evaluate()
            
            if cost < best_swap_cost:
                best_swap_cost = cost
                best_pickup_to_close = pickup_to_close
                best_solution = temp_solution
        
        # Aplicar el mejor swap
        if best_pickup_to_close is not None:
            self.current_solution._clone_solution(best_solution)
            self._add_tabu(best_pickup_to_close, "pickup")
            self.logger.debug(f"Swap pickup: cerrado {best_pickup_to_close}, abierto {best_pickup_to_open}")

    def _get_facility_candidates_to_close(self):
        """Obtiene instalaciones candidatas a ser cerradas (basado en distancia media)"""
        def avg_assignment_distance(j):
            if self.problem == "P2":
                clients = [i for i in self.I if self.current_solution.w.get((i, i, j), 0) == 1]
            else:  # P1
                clients = [i for i in self.I if self.current_solution.x.get((i, j), 0) == 1]
            
            if not clients:
                return 0
            return sum(self.current_solution._get_distance_ij(i, j) for i in clients) / len(clients)
        
        # Ordenar por distancia media descendente (peores primero)
        candidates = sorted(self.current_solution.open_facilities, 
                          key=avg_assignment_distance, reverse=True)
        
        # Retornar hasta la mitad + 1 de las peores instalaciones
        return candidates[:max(1, len(candidates) // 2 + 1)]

    def _get_pickup_candidates_to_close(self):
        """Obtiene puntos de recogida candidatos a ser cerrados"""
        def avg_assignment_distance(k):
            clients = []
            for i in self.I:
                if self.problem == "P1" and self.current_solution.z.get((i, k), 0) == 1:
                    clients.append(i)
                elif self.problem == "P2":
                    for j in self.J:
                        if self.current_solution.w.get((i, k, j), 0) == 1:
                            clients.append(i)
                            break
            
            if not clients:
                return 0
            return sum(self.d_kj[k][i] for i in clients) / len(clients)
        
        candidates = sorted(self.current_solution.open_pickups, 
                          key=avg_assignment_distance, reverse=True)
        
        return candidates[:max(1, len(candidates) // 2 + 1)]
    
    def _get_best_non_tabu_facility(self):
        """Encuentra la mejor instalación cerrada no-tabú para abrir"""
        def facility_score(j):
            # Score basado en distancia media a todos los clientes (menor es mejor)
            return sum(self.current_solution._get_distance_ij(i, j) for i in self.I) / len(self.I)
        
        # Filtrar instalaciones cerradas no-tabú y ordenar por score
        candidates = [j for j in self.current_solution.closed_facilities 
                     if not self._is_facility_tabu(j)]
        
        if not candidates:
            return None
        
        candidates.sort(key=facility_score)
        return candidates[0]  # Mejor candidato (menor distancia media)

    def _get_best_non_tabu_pickup(self):
        """Encuentra el mejor punto de recogida cerrado no-tabú para abrir"""
        def pickup_score(k):
            # Score basado en distancia media a instalaciones abiertas
            if not self.current_solution.open_facilities:
                return float('inf')
            
            distances = [self.d_kj[k][j] for j in self.current_solution.open_facilities 
                        if self.d_kj[k][j] != float('inf')]
            return sum(distances) / len(distances) if distances else float('inf')
        
        candidates = [k for k in self.current_solution.closed_pickups 
                     if not self._is_pickup_tabu(k)]
        
        if not candidates:
            return None
        
        candidates.sort(key=pickup_score)
        return candidates[0]
    
    def _long_term_memory(self):
        """Actualiza la memoria a largo plazo"""
        # Actualizar contadores de residencia
        self.RC_install += self.current_solution.y
        self.RC_pickup += self.current_solution.nu
        
        # Actualizar costos promedio
        for j in self.current_solution.open_facilities:
            self.AO_install[j].append(self.current_solution.cost)
        
        for k in self.current_solution.open_pickups:
            self.AO_pickup[k].append(self.current_solution.cost)

    def _long_term_phase(self):
        """Fase de diversificación usando memoria a largo plazo"""
        try:
            # Calcular función greedy para instalaciones
            g_facilities = []
            max_f_install = max(self.RC_install) if max(self.RC_install) > 0 else 1
            
            # Calcular promedios de costos objetivos
            avg_objectives = []
            for j in self.J:
                costs = self.AO_install[j]
                avg_objectives.append(sum(costs) / len(costs) if costs else self.current_solution.cost)
            
            min_q = min(avg_objectives) if avg_objectives else 0
            max_q = max(avg_objectives) if avg_objectives else 1
            diff_q = max_q - min_q if max_q != min_q else 1
            
            # Calcular función greedy g'(j) según el paper
            for j in self.J:
                base_score = self._compute_facility_impact(j)
                frequency_penalty = self.gamma_f * (self.RC_install[j-1] / max_f_install)
                quality_bonus = self.gamma_q * ((max_q - avg_objectives[j-1]) / diff_q)
                
                g_j = base_score - frequency_penalty + quality_bonus
                g_facilities.append((j, g_j))
            
            # Seleccionar mejores instalaciones
            g_facilities.sort(key=lambda x: x[1], reverse=True)
            best_facilities = [fac[0] for fac in g_facilities[:self.p]]
            
            # Repetir proceso para puntos de recogida
            g_pickups = []
            max_f_pickup = max(self.RC_pickup) if max(self.RC_pickup) > 0 else 1
            
            avg_objectives_pickup = []
            for k in self.K:
                costs = self.AO_pickup[k]
                avg_objectives_pickup.append(sum(costs) / len(costs) if costs else self.current_solution.cost)
            
            min_q_pickup = min(avg_objectives_pickup) if avg_objectives_pickup else 0
            max_q_pickup = max(avg_objectives_pickup) if avg_objectives_pickup else 1
            diff_q_pickup = max_q_pickup - min_q_pickup if max_q_pickup != min_q_pickup else 1
            
            for idx_k, k in enumerate(self.K):
                base_score = self._compute_pickup_impact(k, idx_k)
                frequency_penalty = self.gamma_f * (self.RC_pickup[idx_k] / max_f_pickup)
                quality_bonus = self.gamma_q * ((max_q_pickup - avg_objectives_pickup[idx_k]) / diff_q_pickup)
                
                g_k = base_score - frequency_penalty + quality_bonus
                g_pickups.append((k, g_k))
            
            g_pickups.sort(key=lambda x: x[1], reverse=True)
            best_pickups = [pick[0] for pick in g_pickups[:self.t]]
            
            # Aplicar nueva configuración
            self._apply_diversification(best_facilities, best_pickups)
            
        except Exception as e:
            self.logger.error(f"Error en diversificación: {e}")
            # Fallback: perturbación aleatoria
            self._random_perturbation()

    def _compute_facility_impact(self, j):
        """Calcula el impacto de abrir/mantener la instalación j"""
        temp_solution = Solution(self, other_solution=self.current_solution)
        cost_before = temp_solution.cost
        
        # Si está cerrada, calcular impacto de abrirla
        if j not in self.current_solution.open_facilities:
            temp_solution.y[j-1] = 1
        
        cost_after = temp_solution.evaluate()
        return cost_before - cost_after  # Retorna la mejora (positivo = bueno)
    
    def _compute_pickup_impact(self, k, idx):
        """Calcula el impacto de abrir/mantener el punto de recogida k"""
        temp_solution = Solution(self, other_solution=self.current_solution)
        cost_before = temp_solution.cost
        
        # Si está cerrado, calcular impacto de abrirlo
        if k not in self.current_solution.open_pickups:
            temp_solution.nu[idx] = 1
        
        cost_after = temp_solution.evaluate()
        return cost_before - cost_after
    
    def _apply_diversification(self, best_facilities, best_pickups):
        """Aplica la nueva configuración de diversificación"""
        # Determinar cambios necesarios
        facilities_to_close = [j for j in self.current_solution.open_facilities 
                             if j not in best_facilities]
        facilities_to_open = [j for j in best_facilities 
                            if j not in self.current_solution.open_facilities]
        
        pickups_to_close = [k for k in self.current_solution.open_pickups 
                          if k not in best_pickups]
        pickups_to_open = [k for k in best_pickups 
                         if k not in self.current_solution.open_pickups]
        
        # Reinicializar solución
        self.current_solution._initialize_empty()
        
        # Activar mejores instalaciones
        for j in best_facilities:
            self.current_solution.y[j-1] = 1
        
        # Activar mejores puntos de recogida
        for k in best_pickups:
            idx = self.K.index(k)
            self.current_solution.nu[idx] = 1
        
        # Añadir elementos cerrados a la lista tabú
        for j in facilities_to_close:
            self._add_tabu(j, "facility")
        for k in pickups_to_close:
            self._add_tabu(k, "pickup")
        
        # Evaluar nueva solución
        self.current_solution.evaluate()
        
        self.logger.info(f"Diversificación aplicada. Nuevo costo: {self.current_solution.cost}")

    def _random_perturbation(self):
        """Perturbación aleatoria como fallback"""
        try:
            # Cerrar algunas instalaciones aleatorias y abrir otras
            if len(self.current_solution.open_facilities) > 1:
                to_close = random.choice(self.current_solution.open_facilities)
                to_open = random.choice(self.current_solution.closed_facilities)
                
                self.current_solution.y[to_close-1] = 0
                self.current_solution.y[to_open-1] = 1
                self._add_tabu(to_close, "facility")
            
            # Hacer lo mismo con puntos de recogida
            if len(self.current_solution.open_pickups) > 1:
                to_close = random.choice(self.current_solution.open_pickups)
                to_open = random.choice(self.current_solution.closed_pickups)
                
                idx_close = self.K.index(to_close)
                idx_open = self.K.index(to_open)
                
                self.current_solution.nu[idx_close] = 0
                self.current_solution.nu[idx_open] = 1
                self._add_tabu(to_close, "pickup")
            
            self.current_solution.evaluate()
            self.logger.info(f"Perturbación aleatoria aplicada. Costo: {self.current_solution.cost}")
            
        except Exception as e:
            self.logger.error(f"Error en perturbación aleatoria: {e}")

    def _add_tabu(self, element, move_type):
        """Añade elemento a la lista tabú correspondiente"""
        if move_type == "facility":
            self.tabu_list_facilities.append(element)
        elif move_type == "pickup":
            self.tabu_list_pickup.append(element)

    def _is_facility_tabu(self, facility):
        """Verifica si una instalación está en la lista tabú"""
        return facility in self.tabu_list_facilities

    def _is_pickup_tabu(self, pickup):
        """Verifica si un punto de recogida está en la lista tabú"""
        return pickup in self.tabu_list_pickup

    def _save_best_sol(self, solve_time):
        """Guarda la mejor solución encontrada"""
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        
        filename = f"{self.instance+1}_tabu_best_solution.txt"
        filepath = os.path.join(self.result_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                f.write(f"Instancia: {self.instance+1}\n")
                f.write(f"Problema: {self.problem}\n")
                f.write(f"Inicialización: {self.inicialization}\n")
                f.write(f"Costo mejor solución: {self.best_solution.cost:.2f}\n")
                f.write(f"Tiempo ejecución (seg): {solve_time:.2f}\n")
                f.write(f"Iteraciones máximas: {self.max_iter}\n")
                f.write(f"Tabu tenure instalaciones: {self.tabu_tenure_facilities}\n")
                f.write(f"Tabu tenure pickups: {self.tabu_tenure_pickups}\n")
                f.write("\nInstalaciones abiertas:\n")
                for j in sorted(self.best_solution.open_facilities):
                    f.write(f"  Instalación {j}\n")
                
                f.write("\nPuntos de recogida abiertos:\n")
                for k in sorted(self.best_solution.open_pickups):
                    f.write(f"  Punto {k}\n")
                
                # Información adicional de la solución
                f.write(f"\nTotal instalaciones abiertas: {len(self.best_solution.open_facilities)}/{self.p}\n")
                f.write(f"Total puntos de recogida abiertos: {len(self.best_solution.open_pickups)}/{self.t}\n")
                
        except Exception as e:
            self.logger.error(f"Error al guardar solución: {e}")

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

