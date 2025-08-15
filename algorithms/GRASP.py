import random
import time
import os

from structure.solution import Solution

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
        """
        Búsqueda local según metodología GRASP
        
        Implementa estrategia first-improving como recomienda el paper:
        "In practice, we observed on many applications that quite often both strategies 
        lead to the same final solution, but in smaller computation times when the 
        first-improving strategy is used."
        """
        current = Solution(self, other_solution=solution)
        current.evaluate()
        
        self.logger.info(f"Iniciando búsqueda local desde costo: {current.cost}")
        
        improved = True
        iteration = 0
        
        while improved:
            improved = False
            iteration += 1
            
            self.logger.debug(f"Iteración de búsqueda local: {iteration}")
            
            # Explorar vecindarios en orden: primero instalaciones, luego puntos de recogida
            # Esto sigue el principio de explorar vecindarios simples primero
            
            # Vecindario 1: Intercambio de instalaciones
            neighbor = self._explore_facility_neighborhood(current)
            if neighbor and neighbor.cost < current.cost:
                current = neighbor
                improved = True
                self.logger.debug(f"Mejora encontrada (instalaciones): {current.cost}")
                continue  # First-improving: tomar primera mejora encontrada
            
            # Vecindario 2: Intercambio de puntos de recogida
            neighbor = self._explore_pickup_neighborhood(current)
            if neighbor and neighbor.cost < current.cost:
                current = neighbor
                improved = True
                self.logger.debug(f"Mejora encontrada (puntos recogida): {current.cost}")
                continue  # First-improving: tomar primera mejora encontrada
            
            # Vecindario 3: Intercambio combinado (más costoso, solo si otros fallan)
            neighbor = self._explore_combined_neighborhood(current)
            if neighbor and neighbor.cost < current.cost:
                current = neighbor
                improved = True
                self.logger.debug(f"Mejora encontrada (combinado): {current.cost}")
        
        self.logger.info(f"Búsqueda local terminada después de {iteration} iteraciones. Costo final: {current.cost}")
        return current

    def _explore_facility_neighborhood(self, solution):
        """
        Explora el vecindario de intercambio de instalaciones (1-1 swap)
        Implementa first-improving strategy
        """
        if not solution.open_facilities or not solution.closed_facilities:
            return None
        
        # Ordenar instalaciones para exploración sistemática
        open_facilities = sorted(solution.open_facilities)
        closed_facilities = sorted(solution.closed_facilities)
        
        for facility_out in open_facilities:
            for facility_in in closed_facilities:
                # Crear vecino con intercambio 1-1
                neighbor = Solution(self, other_solution=solution)
                neighbor.y[facility_out-1] = 0 # Cerrar facilidad
                neighbor.y[facility_in-1] = 1   # Abrir facilidad
                
                # Reasignar clientes y evaluar
                neighbor._assign_customers()
                neighbor.evaluate()
                
                # First-improving: devolver primera mejora encontrada
                if neighbor.cost < solution.cost:
                    self.logger.debug(f"Intercambio facilidades: {facility_out} -> {facility_in}, "
                                            f"mejora: {solution.cost - neighbor.cost:.2f}")
                    return neighbor
        
        return None

    def _explore_pickup_neighborhood(self, solution):
        """
        Explora el vecindario de intercambio de puntos de recogida (1-1 swap)
        Implementa first-improving strategy
        """
        if not solution.open_pickups or not solution.closed_pickups:
            return None
        
        # Ordenar puntos de recogida para exploración sistemática
        open_pickups = sorted(solution.open_pickups)
        closed_pickups = sorted(solution.closed_pickups)
        
        for pickup_out in open_pickups:
            for pickup_in in closed_pickups:
                # Crear vecino con intercambio 1-1
                neighbor = Solution(self, other_solution=solution)
                neighbor.nu[self.K.index(pickup_out)] = 0  # Cerrar punto
                neighbor.nu[self.K.index(pickup_in)] = 1   # Abrir punto
                
                # Reasignar clientes y evaluar
                neighbor._assign_customers()
                neighbor.evaluate()
                
                # First-improving: devolver primera mejora encontrada
                if neighbor.cost < solution.cost:
                    self.logger.debug(f"Intercambio puntos recogida: {pickup_out} -> {pickup_in}, "
                                            f"mejora: {solution.cost - neighbor.cost:.2f}")
                    return neighbor
        
        return None

    def _explore_combined_neighborhood(self, solution):
        """
        Explora vecindario combinado: intercambio simultáneo de instalación y punto de recogida
        Solo se usa si los vecindarios simples no encuentran mejoras
        """
        if (not solution.open_facilities or not solution.closed_facilities or 
            not solution.open_pickups or not solution.closed_pickups):
            return None
        
        # Limitar búsqueda combinada para evitar explosión combinatoria
        max_facility_swaps = min(3, len(solution.open_facilities), len(solution.closed_facilities))
        max_pickup_swaps = min(3, len(solution.open_pickups), len(solution.closed_pickups))
        
        open_facilities = random.sample(solution.open_facilities, max_facility_swaps)
        closed_facilities = random.sample(solution.closed_facilities, max_facility_swaps)
        open_pickups = random.sample(solution.open_pickups, max_pickup_swaps)
        closed_pickups = random.sample(solution.closed_pickups, max_pickup_swaps)
        
        for facility_out in open_facilities:
            for facility_in in closed_facilities:
                for pickup_out in open_pickups:
                    for pickup_in in closed_pickups:
                        # Crear vecino con intercambio combinado
                        neighbor = Solution(self, other_solution=solution)
                        neighbor.y[facility_out-1] = 0
                        neighbor.y[facility_in-1] = 1
                        neighbor.nu[self.K.index(pickup_out)] = 0
                        neighbor.nu[self.K.index(pickup_in)] = 1
                        
                        # Reasignar clientes y evaluar
                        neighbor._assign_customers()
                        neighbor.evaluate()
                        
                        # First-improving: devolver primera mejora encontrada
                        if neighbor.cost < solution.cost:
                            self.logger.debug(f"Intercambio combinado: F({facility_out}->{facility_in}), "
                                                    f"P({pickup_out}->{pickup_in}), "
                                                    f"mejora: {solution.cost - neighbor.cost:.2f}")
                            return neighbor
        
        return None

    def generate_neighbors_systematic(self, solution, max_neighbors=None):
        """
        Generación sistemática de vecinos para análisis completo del vecindario
        Útil para debugging o cuando se necesita exploración exhaustiva
        """
        neighbors = []
        
        # Vecinos por intercambio de instalaciones
        if solution.open_facilities and solution.closed_facilities:
            for facility_out in solution.open_facilities:
                for facility_in in solution.closed_facilities:
                    neighbor = Solution(self, other_solution=solution)
                    neighbor.y[facility_out-1] = 0
                    neighbor.y[facility_in-1] = 1
                    neighbor._assign_customers()
                    neighbors.append(neighbor)
                    
                    if max_neighbors and len(neighbors) >= max_neighbors:
                        return neighbors
        
        # Vecinos por intercambio de puntos de recogida
        if solution.open_pickups and solution.closed_pickups:
            for pickup_out in solution.open_pickups:
                for pickup_in in solution.closed_pickups:
                    neighbor = Solution(self, other_solution=solution)
                    neighbor.nu[self.K.index(pickup_out)] = 0
                    neighbor.nu[self.K.index(pickup_in)] = 1
                    neighbor._assign_customers()
                    neighbors.append(neighbor)
                    
                    if max_neighbors and len(neighbors) >= max_neighbors:
                        return neighbors
        
        return neighbors

    def local_search_best_improving(self, solution):
        """
        Variante de búsqueda local con estrategia best-improving
        Útil para comparación o cuando se prefiere calidad sobre velocidad
        """
        current = Solution(self, other_solution=solution)
        current.evaluate()
        
        improved = True
        iteration = 0
        
        while improved:
            improved = False
            iteration += 1
            best_neighbor = None
            best_cost = current.cost
            
            # Generar todos los vecinos del vecindario actual
            all_neighbors = []
            
            # Vecindario de instalaciones
            facility_neighbors = []
            if current.open_facilities and current.closed_facilities:
                for facility_out in current.open_facilities:
                    for facility_in in current.closed_facilities:
                        neighbor = Solution(self, other_solution=current)
                        neighbor.y[facility_out-1] = 0
                        neighbor.y[facility_in-1] = 1
                        neighbor._assign_customers()
                        neighbor.evaluate()
                        facility_neighbors.append(neighbor)
            
            all_neighbors.extend(facility_neighbors)
            
            # Vecindario de puntos de recogida
            pickup_neighbors = []
            if current.open_pickups and current.closed_pickups:
                for pickup_out in current.open_pickups:
                    for pickup_in in current.closed_pickups:
                        neighbor = Solution(self, other_solution=current)
                        neighbor.nu[self.K.index(pickup_out)] = 0
                        neighbor.nu[self.K.index(pickup_in)] = 1
                        neighbor._assign_customers()
                        neighbor.evaluate()
                        pickup_neighbors.append(neighbor)
            
            all_neighbors.extend(pickup_neighbors)
            
            # Encontrar el mejor vecino
            for neighbor in all_neighbors:
                if neighbor.cost < best_cost:
                    best_cost = neighbor.cost
                    best_neighbor = neighbor
            
            # Actualizar si se encontró mejora
            if best_neighbor:
                current = best_neighbor
                improved = True
                self.logger.debug(f"Mejor mejora encontrada: {best_cost}")
        
        return current
    
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