import logging
import random
import time
import os

import numpy as np
from structure.solution import Solution

class GRASPSearch:
    def __init__(self, params, instance, result_dir, logger: logging, frac_neighbors: int = 4, 
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
        self.n_pickups = len(params['K'])
        self.n_customers = len(self.I)
        
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
        
        # Pre-computar estructuras de datos para optimización
        self._precompute_structures()
    
    def _precompute_structures(self):
        """Pre-computa estructuras de datos para optimizar la búsqueda local"""
        # Índices pre-computados para acceso rápido
        self.pickup_to_index = {k: idx for idx, k in enumerate(self.K)}
        
        # Límites para controlar explosión combinatoria
        self.max_facility_moves = min(20, len(self.J))
        self.max_pickup_moves = min(20, len(self.K))
        
        # Cache para evaluaciones de vecindario
        self.evaluation_cache = {}
    
    def run(self):
        """Ejecución principal del algoritmo GRASP mejorado"""
        start_time = time.time()
        self.best_solution = None
        consecutive_no_improvement = 0
        max_consecutive = 10  # Criterio de parada temprana
        
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
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1
            
            # Criterio de parada temprana si no hay mejoras
            if consecutive_no_improvement >= max_consecutive:
                self.logger.info(f"Parada temprana después de {max_consecutive} iteraciones sin mejora")
                break
            
            elapsed = time.time() - start_time
            self.best_solution.time = elapsed
            self.logger.info(f"Tiempo transcurrido: {elapsed:.2f} segundos")
        
        elapsed = time.time() - start_time
        self.logger.info(f"Mejor costo encontrado: {self.best_solution.cost}")
        self._save_best_sol(solve_time=elapsed)
    
    def local_search(self, solution):
        """
        Búsqueda local mejorada basada en el paper GRASP para GDP
        
        Implementa mejoras clave del paper:
        1. Lista pivotal para identificar elementos críticos
        2. First-improving con exploración eficiente
        3. Evaluación
        4. Filtrado de movimientos no prometedores
        """
        current = Solution(self, other_solution=solution)
        current.evaluate()
        
        self.logger.debug(f"Iniciando búsqueda local mejorada desde costo: {current.cost}")
        
        improved = True
        iteration = 0
        
        while improved:
            improved = False
            iteration += 1
            
            self.logger.debug(f"Iteración de búsqueda local: {iteration}")
            
            # Crear lista pivotal: identificar elementos que más contribuyen al costo
            pivotal_facilities, pivotal_pickups = self._create_pivotal_lists(current)
            
            # Explorar vecindarios con priorización
            # Vecindario 1: Intercambio de instalaciones pivotales
            neighbor = self._explore_pivotal_facility_neighborhood(current, pivotal_facilities)
            if neighbor and neighbor.cost < current.cost:
                current = neighbor
                improved = True
                self.logger.debug(f"Mejora encontrada (instalaciones pivotales): {current.cost}")
                continue
            
            # Vecindario 2: Intercambio de puntos de recogida pivotales
            neighbor = self._explore_pivotal_pickup_neighborhood(current, pivotal_pickups)
            if neighbor and neighbor.cost < current.cost:
                current = neighbor
                improved = True
                self.logger.debug(f"Mejora encontrada (puntos recogida pivotales): {current.cost}")
                continue
            
            # Vecindario 3: Intercambio general si pivotales fallan
            neighbor = self._explore_general_facility_neighborhood(current)
            if neighbor and neighbor.cost < current.cost:
                current = neighbor
                improved = True
                self.logger.debug(f"Mejora encontrada (instalaciones generales): {current.cost}")
                continue
                
            neighbor = self._explore_general_pickup_neighborhood(current)
            if neighbor and neighbor.cost < current.cost:
                current = neighbor
                improved = True
                self.logger.debug(f"Mejora encontrada (puntos recogida generales): {current.cost}")
        
        self.logger.info(f"Búsqueda local terminada después de {iteration} iteraciones. Costo final: {current.cost}")
        return current
    
    def _create_pivotal_lists(self, solution: Solution):
        """
        Crea listas pivotales identificando elementos que más contribuyen al costo
        """
        pivotal_facilities = []
        pivotal_pickups = []
        
        # Calcular contribución de cada instalación abierta al costo total
        facility_contributions = {}
        
        for facility in solution.open_facilities:
            total_cost = 0
            assigned_customers = []

            # Usar un generador para encontrar clientes asignados y almacenarlos en una lista
            for customer in self.I:
                if self.problem == "P1":
                    if solution.x.get((customer, facility), 0) == 1:
                        assigned_customers.append(customer)
                else:
                    if solution.w.get((customer, customer, facility), 0) == 1:
                        assigned_customers.append(customer)

            # Si hay clientes asignados, calcular el costo total y luego el promedio
            if assigned_customers:
                num_assigned_customers = len(assigned_customers)
                for customer in assigned_customers:
                    total_cost += solution._get_distance_ij(customer, facility)
                
                # Calcular el costo promedio y asignarlo
                facility_contributions[facility] = total_cost / num_assigned_customers
            else:
                # Si no hay clientes asignados, el costo es 0
                facility_contributions[facility] = 0
        
        pickup_contributions = {pickup: 0 for pickup in solution.open_pickups}

        for pickup in solution.open_pickups:
            total_cost = 0
            assigned_facilities = []
            # Usar un generador para encontrar instalaciones asignadas, evitando iterar sobre todas las instalaciones
            if self.problem == "P1":
                for facility in solution.open_facilities: 
                    if solution.s.get((pickup, facility)) > 0:
                        assigned_facilities.append(facility)
            elif self.problem == "P2":
                for facility in solution.open_facilities: 
                    if any(solution.w.get((i, pickup, facility), 0) == 1 for i in self.I):
                        assigned_facilities.append(facility)

            if assigned_facilities:
                num_assigned_facilities = len(assigned_facilities)
                for facility in assigned_facilities:
                    total_cost += self.d_kj[pickup][facility]
            # Calcular el costo promedio y asignarlo
                pickup_contributions[pickup] = total_cost / num_assigned_facilities
            else:
                # Si no hay clientes asignados, el costo es 0
                pickup_contributions[pickup] = 0
        
        # Seleccionar elementos pivotales (los de mayor contribución)
        if facility_contributions:
            sorted_facilities = sorted(facility_contributions.items(), 
                                     key=lambda x: x[1], reverse=True)
            # Tomar hasta el 50% de instalaciones con mayor contribución
            n_pivotal = max(1, len(sorted_facilities) // 2)
            pivotal_facilities = [f[0] for f in sorted_facilities[:n_pivotal]]
        
        if pickup_contributions:
            sorted_pickups = sorted(pickup_contributions.items(),
                                  key=lambda x: x[1], reverse=True)
            n_pivotal = max(1, len(sorted_pickups) // 2)
            pivotal_pickups = [p[0] for p in sorted_pickups[:n_pivotal]]
        
        self.logger.debug(f"Instalaciones pivotales: {pivotal_facilities}")
        self.logger.debug(f"Puntos recogida pivotales: {pivotal_pickups}")
        
        return pivotal_facilities, pivotal_pickups
    
    def _explore_pivotal_facility_neighborhood(self, solution: Solution, pivotal_facilities):
        """
        Explora intercambios priorizando instalaciones pivotales
        """
        if not pivotal_facilities or not solution.closed_facilities:
            return None
        
        # Priorizar intercambio de instalaciones pivotales por cerradas con mejor ratio
        closed_facilities_sorted = self._sort_closed_facilities_by_potential(solution)
        
        for facility_out in pivotal_facilities:
            for facility_in in closed_facilities_sorted[:self.max_facility_moves]:
                # Verificar si el intercambio es prometedor antes de evaluar completamente
                neighbor = self._create_facility_swap_neighbor(solution, facility_out, facility_in)
                if neighbor and neighbor.cost < solution.cost:
                    return neighbor
        
        return None
    
    def _explore_pivotal_pickup_neighborhood(self, solution: Solution, pivotal_pickups):
        """
        Explora intercambios priorizando puntos de recogida pivotales
        """
        if not pivotal_pickups or not solution.closed_pickups:
            return None
        
        closed_pickups_sorted = self._sort_closed_pickups_by_potential(solution)
        
        for pickup_out in pivotal_pickups:
            for pickup_in in closed_pickups_sorted[:self.max_pickup_moves]:
                neighbor = self._create_pickup_swap_neighbor(solution, pickup_out, pickup_in)
                if neighbor and neighbor.cost < solution.cost:
                    return neighbor
        
        return None
    
    def _explore_general_facility_neighborhood(self, solution: Solution):
        """Exploración general del vecindario de instalaciones si pivotales fallan"""
        if not solution.open_facilities or not solution.closed_facilities:
            return None
        
        # Muestreo aleatorio para evitar exploración exhaustiva
        open_sample = random.sample(solution.open_facilities, 
                                   min(self.max_facility_moves, len(solution.open_facilities)))
        closed_sample = random.sample(solution.closed_facilities,
                                     min(self.max_facility_moves, len(solution.closed_facilities)))
        
        for facility_out in open_sample:
            for facility_in in closed_sample:
                neighbor = self._create_facility_swap_neighbor(solution, facility_out, facility_in)
                if neighbor and neighbor.cost < solution.cost:
                    return neighbor
        
        return None
    
    def _explore_general_pickup_neighborhood(self, solution: Solution):
        """Exploración general del vecindario de puntos de recogida"""
        if not solution.open_pickups or not solution.closed_pickups:
            return None
        
        open_sample = random.sample(solution.open_pickups,
                                   min(self.max_pickup_moves, len(solution.open_pickups)))
        closed_sample = random.sample(solution.closed_pickups,
                                     min(self.max_pickup_moves, len(solution.closed_pickups)))
        
        for pickup_out in open_sample:
            for pickup_in in closed_sample:
                neighbor = self._create_pickup_swap_neighbor(solution, pickup_out, pickup_in)
                if neighbor and neighbor.cost < solution.cost:
                    return neighbor
        
        return None
    
    def _sort_closed_facilities_by_potential(self, solution: Solution):
        """Ordena instalaciones cerradas por su potencial de mejora"""
        potentials = []
        for facility in solution.closed_facilities:
            # Agregar factor de ubicación: proximidad promedio a clientes
            avg_distance = sum(solution._get_distance_ij(i, facility) for i in self.I) / len(self.I)
            
            potentials.append((facility, avg_distance))
        
        # Ordenar por potencial (mayor potencial primero)
        return [f[0] for f in sorted(potentials, key=lambda x: x[1], reverse=True)]
    
    def _sort_closed_pickups_by_potential(self, solution: Solution):
        """Ordena puntos de recogida cerrados por su potencial de mejora"""
        potentials = []
        for pickup in solution.closed_pickups:
            # Agregar factor de ubicación
            avg_distance = sum(self.d_kj[pickup][j] for j in self.J) / len(self.J)
            potentials.append((pickup, avg_distance))
        
        return [p[0] for p in sorted(potentials, key=lambda x: x[1], reverse=True)]
    
    def _create_facility_swap_neighbor(self, solution, facility_out, facility_in):
        """Crea vecino optimizado para intercambio de instalaciones"""
        try:
            neighbor = Solution(self, other_solution=solution)
            neighbor.y[facility_out - 1] = 0
            neighbor.y[facility_in - 1] = 1
            
            neighbor.evaluate()
            
            return neighbor
        except Exception as e:
            self.logger.warning(f"Error creando vecino de instalación: {e}")
            return None
    
    def _create_pickup_swap_neighbor(self, solution, pickup_out, pickup_in):
        """Crea vecino optimizado para intercambio de puntos de recogida"""
        try:
            neighbor = Solution(self, other_solution=solution)
            neighbor.nu[self.pickup_to_index[pickup_out]] = 0
            neighbor.nu[self.pickup_to_index[pickup_in]] = 1
            
            neighbor.evaluate()
            
            return neighbor
        except Exception as e:
            self.logger.warning(f"Error creando vecino de punto recogida: {e}")
            return None
    
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