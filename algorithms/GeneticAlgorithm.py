import os
import time
import traceback
import numpy as np
import random
from typing import List, Tuple

class Solution:
    def __init__(self, genetic_algorithm, other_solution=None):
        self.genetic_algorithm = genetic_algorithm
        self.problem = genetic_algorithm.problem
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
        return [k+1+self.genetic_algorithm.n_customers for k, val in enumerate(self.nu) if val == 1]
    
    @property
    def closed_pickups(self):
        return [k+1+self.genetic_algorithm.n_customers for k, val in enumerate(self.nu) if val == 0]
    
    def _initialize_empty(self):
        """Inicialización optimizada según el tipo de problema"""
        self.y = np.zeros(self.genetic_algorithm.n_customers, dtype=int)
        self.nu = np.zeros(self.genetic_algorithm.n_pickups, dtype=int)
        
        if self.problem == "P1":
            # Variables x_ij: 1 si cliente i es asignado directamente a facilidad j
            self.x = {(i, j): 0 for i in self.genetic_algorithm.I for j in self.genetic_algorithm.J}
            # Variables z_ik: 1 si cliente i va al punto de recogida k
            self.z = {(i, k): 0 for i in self.genetic_algorithm.I for k in self.genetic_algorithm.K_i[i]}
            # Variables s_kj: flujo de demanda del punto k a la facilidad j
            self.s = {(k, j): 0.0 for k in self.genetic_algorithm.K for j in self.genetic_algorithm.J}
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
            return self.genetic_algorithm.d_ij[i][j]
        else:
            return self.genetic_algorithm.d_ij[j][i]
    
    def initialize_greedy(self, alpha=0.3):
        """Construcción greedy aleatorizada optimizada"""
        try:
            # Reset de variables de decisión
            self._initialize_empty()
            
            # --- Selección de instalaciones ---
            facility_scores = [
                (j, sum(self._get_distance_ij(i, j) for i in self.genetic_algorithm.I) / self.genetic_algorithm.n_customers)
                for j in self.genetic_algorithm.J
            ]
            facility_scores.sort(key=lambda x: x[1])
            rcl_size = max(1, int(alpha * len(facility_scores)))
            selected_facilities = set()
            
            for _ in range(self.genetic_algorithm.p):
                # Construcción del RCL directamente filtrando ya los seleccionados
                rcl = [pair for pair in facility_scores if pair[0] not in selected_facilities][:rcl_size]
                if not rcl: break  # Seguridad por si faltan elementos
                j, _ = random.choice(rcl)
                self.y[j-1] = 1
                selected_facilities.add(j)
            
            self.genetic_algorithm.logger.info(f"[Genetic Algorithm] Instalaciones seleccionadas: {[j for j in selected_facilities]}")
            
            # --- Selección de puntos de recogida ---
            pickup_scores = []
            for k in self.genetic_algorithm.K:
                dists = [self.genetic_algorithm.d_kj[k][j] for j in self.open_facilities if self.genetic_algorithm.d_kj[k][j] != float('inf')]
                avg_dist = sum(dists) / len(dists) if dists else float('inf')
                pickup_scores.append((k, avg_dist))
            
            pickup_scores.sort(key=lambda x: x[1])
            rcl_size_pickup = max(1, int(alpha * len(pickup_scores)))
            selected_pickups = set()
            
            for _ in range(self.genetic_algorithm.t):
                rcl = [pair for pair in pickup_scores if pair[0] not in selected_pickups][:rcl_size_pickup]
                if not rcl: break
                k, _ = random.choice(rcl)
                self.nu[k-self.genetic_algorithm.n_customers-1] = 1
                selected_pickups.add(k)
            
            self.genetic_algorithm.logger.info(f"[Genetic Algorithm] Puntos de recogida seleccionados: {[k for k in selected_pickups]}")
        except Exception as e:
            self.genetic_algorithm.logger.error(f"Error en initialize_greedy: {e}")
            self.genetic_algorithm.logger.error(traceback.format_exc())
            raise
    
    def initialize_random(self):
        try:
            for j in random.sample(self.genetic_algorithm.J, self.genetic_algorithm.p):
                self.y[j-1] = 1
            self.genetic_algorithm.logger.debug(f"Instalaciones seleccionadas: {[j for j in self.genetic_algorithm.J if self.y[j-1] == 1]}")
            
            for k in random.sample(self.genetic_algorithm.K, self.genetic_algorithm.t):
                self.nu[k-self.genetic_algorithm.n_customers-1] = 1
            self.genetic_algorithm.logger.debug(f"Puntos de recogida seleccionados: {[k for k in self.genetic_algorithm.K if self.nu[k-self.genetic_algorithm.n_customers-1] == 1]}")
        except Exception as e:
            self.genetic_algorithm.logger.error(f"Error en initialize_random: {e}")
            self.genetic_algorithm.logger.error(traceback.format_exc())
            raise
    
    def evaluate(self):
        try:
            self.genetic_algorithm.logger.debug("Asignando clientes...")
            self._assign_customers()
            self.genetic_algorithm.logger.debug("Calculando costos...")
            
            if self.problem == "P1":
                cost = self._calculate_cost_P1()
            else:
                cost = self._calculate_cost_P2()
            
            self.cost = cost
            self.genetic_algorithm.logger.info(f"Costo total calculado: {self.cost}")
            return cost
        
        except Exception as e:
            self.genetic_algorithm.logger.error(f"Error en evaluate: {e}")
            self.genetic_algorithm.logger.error(traceback.format_exc()) # Añade esta línea
            self.cost = float('inf')
            return self.cost
    
    def _assign_customers(self):
        """Asignación de clientes optimizada según el modelo P1"""
        
        self.genetic_algorithm.logger.debug(f"Instalaciones abiertas: {self.open_facilities}")
        self.genetic_algorithm.logger.debug(f"Puntos de recogida abiertos: {self.open_pickups}")
        
        for i in self.genetic_algorithm.I:
            try:
                # Limpieza optimizada de asignaciones previas
                self._clear_client_assignments(i)
                demand = self.genetic_algorithm.h[i-1]
                
                self.genetic_algorithm.logger.debug(f"Reasignando cliente {i}...")
                
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
                available_pickups = [k for k in self.open_pickups if k in self.genetic_algorithm.K_i[i]]
                
                for k in available_pickups:
                    # Encontrar la mejor facilidad para servir este punto de recogida
                    best_facility_for_k = min(self.open_facilities, key=lambda j: self.genetic_algorithm.d_kj[k][j])
                    cost = self.genetic_algorithm.d_kj[k][best_facility_for_k]
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_assignment = ('pickup', k, best_facility_for_k)
                
                # Ejecutar la mejor asignación
                if best_assignment[0] == 'direct':
                    j = best_assignment[1]
                    if self.genetic_algorithm.problem == "P1":
                        self.x[(i, j)] = 1
                    else:
                        self.w[(i, i, j)] = 1
                    self.genetic_algorithm.logger.debug(f"Cliente {i} asignado directamente a facilidad {j}")
                else:  # pickup
                    k, j = best_assignment[1], best_assignment[2]
                    if self.genetic_algorithm.problem == "P1":
                        self.z[(i, k)] = 1
                        self.s[(k, j)] += demand
                    else:
                        self.w[(i, k, j)] = 1
                    self.genetic_algorithm.logger.debug(f"Cliente {i} asignado a punto {k}, servido por facilidad {j}")
            except Exception as e:
                self.genetic_algorithm.logger.error(f"Error al asignar cliente {i}: {e}")
                self.genetic_algorithm.logger.error(traceback.format_exc())
                raise
    
    def _clear_client_assignments(self, i):
        """Limpia las asignaciones previas de un cliente"""
        try:
            if self.problem == "P1":
                # Limpiar x del cliente i
                for j in self.genetic_algorithm.J:
                    self.x[(i, j)] = 0
                
                # Limpiar z del cliente i y actualizar s correspondiente
                for k in self.genetic_algorithm.K_i[i]:
                    if (i, k) in self.z and self.z[(i, k)] == 1:
                        demand = self.genetic_algorithm.h[i-1]
                        # Encontrar qué facilidad servía este punto y reducir el flujo
                        for j in self.genetic_algorithm.J:
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
            self.genetic_algorithm.logger.error(f"Error al limpiar asignaciones del cliente {i}: {e}")
            self.genetic_algorithm.logger.error(traceback.format_exc())
            raise
    
    def _calculate_cost_P1(self):
        """Cálculo del costo según la formulación P1 del paper"""
        try:
            cost = 0
            
            # Primer término: Σ(i∈I) Σ(j∈J) h_i * d_ij * x_ij
            # Coste por asignaciones directas cliente → instalación
            for (i, j), value in self.x.items():
                if value == 1:
                    cost += self.genetic_algorithm.h[i-1] * self._get_distance_ij(i, j)
                    self.genetic_algorithm.logger.debug(f"Cliente {i} → Facilidad {j}: {self.genetic_algorithm.h[i-1]} * {self._get_distance_ij(i, j)} = {self.genetic_algorithm.h[i-1] * self._get_distance_ij(i, j)}")
            
            # Segundo término: Σ(k∈K) Σ(j∈J) d_kj * s_kj
            # Coste por transporte desde facilidades a puntos de recogida
            for (k, j), flow in self.s.items():
                if flow > 0:
                    transport_cost = self.genetic_algorithm.d_kj[k][j] * flow
                    cost += transport_cost
                    self.genetic_algorithm.logger.debug(f"Punto {k} ← Facilidad {j}: {flow} * {self.genetic_algorithm.d_kj[k][j]} = {transport_cost}")
            
            return cost
        except Exception as e:
            self.genetic_algorithm.logger.error(f"Error en cálculo de costo P1: {e}")
            self.genetic_algorithm.logger.error(traceback.format_exc())
            raise
    
    def _calculate_cost_P2(self):
        try:
            cost = 0
            for (i, k, j), value in self.w.items():
                if value == 1:
                    if k == i:
                        cost += self.genetic_algorithm.h[i-1] * self._get_distance_ij(i, j)
                    else:
                        cost += self.genetic_algorithm.h[i-1] * self.genetic_algorithm.d_kj[k][j]
            return cost
        except Exception as e:
            self.genetic_algorithm.logger.error(f"Error en cálculo de costo P2: {e}")
            self.genetic_algorithm.logger.error(traceback.format_exc())
            raise

class GeneticSearch:
    """
    Algoritmo Genético para resolver el problema Close-enough Facility Location
    
    El problema consiste en encontrar las mejores ubicaciones para facilidades
    donde cada cliente puede ser servido por cualquier facilidad que esté
    dentro de su "radio de tolerancia".
    """
    def __init__(
                 self, 
                 params, 
                 instance, 
                 result_dir, 
                 logger, 
                 inicializacion: str = "random",
                 generations: int = 500,
                 tournament: int = 10,
                 mutation_rate: float = 0.05,
                 crossover_rate: float = 0.95,
                 problem: str = "P2", 
                ):
        self.I = params['I']
        self.J = params['J']
        self.n_customers = len(self.I)
        self.n_pickups = len(params['K'])
        self.K = params['K']
        self.h = params['h']
        self.d_ij = params['d_ij']
        self.d_kj = params['d_kj']
        self.K_i = params['K_i']
        self.p = params['p']
        self.t = params['t']
        self.service_radii = params.get('R', 5)  # Radio de servicio por defecto
        
        self.best_individual = None
        self.history = []
        self.result_dir = result_dir
        self.instance = instance
        self.problem = problem
        self.logger = logger
        
        # Parámetros del GA
        self.inicialization = inicializacion
        self.population_size = min(10 + self.n_customers // 2, 50)
        self.population = []
        self.tournament = tournament
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Historia de evolución
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def run(self):
        """Ejecuta el algoritmo genético"""
        self.logger.info("Iniciando algoritmo genético...")
        self.logger.info(f"Población: {self.population_size}, Generaciones: {self.generations}")
        self.logger.info(f"Clientes: {self.n_customers}, Facilidades abiertas: {self.p}, Puntos de recogida abiertos: {self.t}")
        
        start_time = time.time()
        # Crear población inicial
        self._create_population()
        best_fitness = float('inf')
        
        for generation in range(self.generations):
            # Evaluar toda la población
            try:
                fitnesses = [ind.evaluate() for ind in self.population]
            except Exception as e:
                self.logger.error(f"Error al evaluar la población: {e}")
                self.logger.error(traceback.format_exc())
                raise
            
            # Actualizar mejor solución
            gen_best_idx = np.argmin(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]
            
            if gen_best_fitness < best_fitness:
                best_fitness = gen_best_fitness
                self.logger.info(f"Mejor individuo generación {generation}: {best_fitness}")
                self.best_individual = Solution(self)
                self.best_individual._clone_solution(self.population[gen_best_idx])
            
            # Evolucionar población
            self._evolve_population(fitnesses)
        
        solve_time = time.time() - start_time
        
        if self.best_individual is None:
            self.logger.warning("Error: No se encontró ninguna solución válida después de todas las iteraciones.")
            return None
        
        self._save_best_sol(solve_time)
        
        print(f"\nOptimización completada!")
        print(f"Mejor fitness: {best_fitness:.2f}")
        print(f"Facilidades seleccionadas: {self.best_individual.open_facilities}")
        print(f"Puntos de recogida seleccionados: {self.best_individual.open_pickups}")
    
    def _evolve_population(self, fitnesses):
        """Evoluciona la población mediante selección, cruzamiento y mutación"""
        # Selección de padres
        parent1 = self._tournament_selection(fitnesses)
        parent2 = self._tournament_selection(fitnesses)
        
        # Cruzamiento para instalaciones
        facilities_child1, facilities_child2 = self._crossover(
            parent1.y, parent2.y, self.n_customers, self.p)
        
        # Cruzamiento para puntos de recogida
        pickups_child1, pickups_child2 = self._crossover(
            parent1.nu, parent2.nu, self.n_pickups, self.t)
        
        # Mutación
        facilities_child1 = self._mutate(facilities_child1, self.p)
        facilities_child2 = self._mutate(facilities_child2, self.p)
        pickups_child1 = self._mutate(pickups_child1, self.t)
        pickups_child2 = self._mutate(pickups_child2, self.t)
        
        # Crear nuevos individuos
        try:
            child1 = Solution(self)
            child1.y = facilities_child1
            child1.nu = pickups_child1
            cost_child1 = child1.evaluate()
        except Exception as e:
            self.logger.error("Error creando o evaluando child1")
            self.logger.error(traceback.format_exc())
            raise
        
        try:
            child2 = Solution(self)
            child2.y = facilities_child2
            child2.nu = pickups_child2
            cost_child2 = child2.evaluate()
        except Exception as e:
            self.logger.error("Error creando o evaluando child2")
            self.logger.error(traceback.format_exc())
            raise
        
        # Reemplazar los peores individuos
        self._replace_worst(fitnesses, [child1, child2], [cost_child1, cost_child2])
    
    def _replace_worst(self, fitnesses, children, children_costs):
        """Reemplaza los peores individuos de la población"""
        sorted_indices = np.argsort(fitnesses)[::-1]  # De peor a mejor
        
        # Obtener costos de todos los candidatos a reemplazo
        all_costs = [fitnesses[sorted_indices[0]], fitnesses[sorted_indices[1]]] + children_costs
        all_individuals = [self.population[sorted_indices[0]], self.population[sorted_indices[1]]] + children
        
        # Ordenar por costo (mejor = menor costo)
        sorted_candidates = sorted(zip(all_costs, all_individuals), key=lambda x: x[0])
        
        # Reemplazar los dos peores con los dos mejores candidatos
        # Primero eliminamos los peores (en orden inverso para no afectar índices)
        indices_to_remove = sorted([sorted_indices[0], sorted_indices[1]], reverse=True)
        for idx in indices_to_remove:
            self.population.pop(idx)
        
        # Agregar los dos mejores candidatos
        self.population.extend([sorted_candidates[0][1], sorted_candidates[1][1]])
    
    def _create_individual(self) -> None:
        """Crea un individuo (solución) aleatoria"""
        cromosoma = Solution(self)
        
        if self.inicialization == "greedy":
            self.logger.debug("Seleccionado inicialización [GREEDY]")
            cromosoma.initialize_greedy()
        elif self.inicialization == "random":
            self.logger.debug("Seleccionado inicialización [RANDOM]")
            cromosoma.initialize_random()
        else:
            raise ValueError(f"Inicialización desconocida: {self.inicialization}")
        
        self.population.append(cromosoma)
    
    def _create_population(self) -> None:
        """Crea la población inicial"""
        for _ in range(self.population_size): 
            self._create_individual()
    
    def _tournament_selection(self, fitnesses: List[float]) -> Solution:
        """Selección por torneo"""
        tournament_size = min(self.tournament, len(self.population))
        tournament = random.sample(list(zip(self.population, fitnesses)), tournament_size)
        winner = min(tournament, key=lambda x: x[1])
        new_ind = Solution(self)
        new_ind._clone_solution(winner[0])
        return new_ind
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray, n_candidates: int, n_open: int) -> Tuple[np.ndarray, np.ndarray]:
        """Cruzamiento de un punto"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Un punto de cruzamiento
        point = random.randint(1, n_candidates - 1)
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Intercambiar desde el punto de cruzamiento
        child1[point:] = parent2[point:]
        child2[point:] = parent1[point:]
        
        # Reparar individuos
        child1 = self._repair_individual(child1, n_open)
        child2 = self._repair_individual(child2, n_open)
        
        return child1, child2
    
    def _mutate(self, individual: np.ndarray, n_open: int) -> np.ndarray:
        """Mutación bit-flip con reparación"""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]  # Flip bit
        
        return self._repair_individual(mutated, n_open)
    
    def _repair_individual(self, individual: np.ndarray, n_open: int) -> np.ndarray:
        """Repara un individuo para que respete la restricción de tener exactamente n_open elementos abiertos."""
        open_positions = np.where(individual == 1)[0]
        closed_positions = np.where(individual == 0)[0]
        num_open = len(open_positions)
        
        if num_open > n_open:
            # Cerrar facilidades/puntos sobrantes
            to_close = random.sample(list(open_positions), num_open - n_open)
            individual[to_close] = 0
        elif num_open < n_open:
            # Abrir facilidades/puntos faltantes
            to_open = random.sample(list(closed_positions), n_open - num_open)
            individual[to_open] = 1
        
        return individual
    
    def _save_best_sol(self, solve_time):
        """Guardado optimizado de la mejor solución"""
        os.makedirs(self.result_dir, exist_ok=True)
        
        filename = f"{self.instance+1}_best_solution.txt"
        filepath = os.path.join(self.result_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(f"Instancia: {self.instance+1}\n")
            f.write(f"Problema: {self.problem}\n")
            f.write(f"Tamaño población: {self.population_size}\n")
            f.write(f"Generaciones: {self.generations}\n")
            f.write(f"Costo mejor solución: {self.best_individual.cost}\n")
            f.write(f"Tiempo ejecución (seg): {solve_time:.2f}\n")
            
            f.write("Instalaciones abiertas:\n")
            for j in self.best_individual.open_facilities:
                f.write(f"  Instalación {j}\n")
            
            f.write("Puntos de recogida abiertos:\n")
            for k in self.best_individual.open_pickups:
                f.write(f"  Punto {k}\n")
            
            if self.problem == "P1":
                f.write("\nAsignaciones directas (x_ij):\n")
                for (i, j), value in self.best_individual.x.items():
                    if value == 1:
                        f.write(f"  Cliente {i} -> Facilidad {j}\n")
                
                f.write("\nAsignaciones a puntos de recogida (z_ik):\n")
                for (i, k), value in self.best_individual.z.items():
                    if value == 1:
                        f.write(f"  Cliente {i} -> Punto {k}\n")
                
                f.write("\nFlujos de puntos a facilidades (s_kj):\n")
                for (k, j), flow in self.best_individual.s.items():
                    if flow > 0:
                        f.write(f"  Punto {k} <- Facilidad {j}: {flow}\n")
        
        self.logger.info(f"Mejor solución guardada en: {filepath}")