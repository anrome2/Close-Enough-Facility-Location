import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
from typing import List, Tuple, Dict
from copy import deepcopy

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
            self.x = {i: 0 for i in self.genetic_algorithm.I}
            self.z = {i: 0 for i in self.genetic_algorithm.J}
            self.s = {(k, j): 0 for k in self.genetic_algorithm.K for j in self.genetic_algorithm.J}
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

    def initialize_random(self):
        for j in random.sample(self.genetic_algorithm.J, self.genetic_algorithm.p):
            self.y[j-1] = 1
        self.genetic_algorithm.logger.debug(f"Instalaciones seleccionadas: {[j for j in self.genetic_algorithm.J if self.y[j-1] == 1]}")

        for k in random.sample(self.genetic_algorithm.K, self.genetic_algorithm.t):
            self.nu[k-self.genetic_algorithm.n_customers-1] = 1
        self.genetic_algorithm.logger.debug(f"Puntos de recogida seleccionados: {[k for k in self.genetic_algorithm.K if self.nu[k-self.genetic_algorithm.n_customers-1] == 1]}")

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
            self.cost = float('inf')
            return self.cost
        
    def _assign_customers(self):
        """Asignación de clientes optimizada"""
        
        self.genetic_algorithm.logger.debug(f"Instalaciones abiertas: {self.open_facilities}")
        self.genetic_algorithm.logger.debug(f"Puntos de recogida abiertos: {self.open_pickups}")
        
        for i in self.genetic_algorithm.I:
            # Limpieza optimizada de asignaciones previas
            self._clear_client_assignments(i)
            demand = self.genetic_algorithm.h[i-1]
            self.genetic_algorithm.logger.debug(f" Reasignando cliente {i}...")

            best_facility = None
            best_pickup = None
            best_cost = float('inf')

            best_facility = min(self.open_facilities, key=lambda j: self._get_distance_ij(i, j))
            best_cost = self._get_distance_ij(i, best_facility)
            if best_cost == 0:
                self._update_assignment(i=i, best_pickup=i, best_facility=best_facility)
            else:
                # Convertimos las listas a conjuntos para una intersección eficiente
                pickup_candidates_set = set(self.genetic_algorithm.K_i[i]) & set(self.open_pickups)
                pickup_candidates = list(pickup_candidates_set)
                for k in pickup_candidates:
                    # Encontrar la mejor facilidad para servir este punto de recogida
                    best_facility_for_k = min(self.open_facilities, key=lambda j: self.genetic_algorithm.d_kj[k][j])
              
                    if self.genetic_algorithm.d_kj[best_pickup][i] < best_cost:
                        self._update_assignment(i=i, best_pickup=best_pickup, demand=demand)
                    else:
                        self._update_assignment(i=i, best_pickup=i, best_facility=best_facility)
                else:
                    self._update_assignment(i=i, best_pickup=i, best_facility=best_facility)
            
    def _clear_client_assignments(self, i):
        """Limpia las asignaciones previas de un cliente"""
        if self.problem == "P1":
            # Limpiar x e z del cliente i
            self.x[i] = 0
            k = self.z[i]
            self.z[i] = 0
            demand = self.genetic_algorithm.h[i-1]
            # Encontrar qué facilidad servía este punto y reducir el flujo
            for j in self.genetic_algorithm.J:
                if self.s[(k, j)] >= demand:
                    self.s[(k, j)] -= demand
                    break
        else:  # P2
            # Limpiar w del cliente i
            keys_to_clear = [key for key in self.w.keys() if key[0] == i]
            for key in keys_to_clear:
                self.w[key] = 0

    def _assign_best_facility(self, pickup):
        """Asigna el mejor facility para un punto de recogida k"""
        return min(self.open_facilities, key=lambda j: self.genetic_algorithm.d_kj[pickup][j])
    
    def _update_assignment(self, i, best_pickup: int = None, best_facility: int = None, demand: float = 0.0):
        """Actualiza las estructuras de datos con la mejor asignación"""
       
        if self.problem == "P1":
            if best_pickup != i:
                self.z[i] = best_pickup
                self.genetic_algorithm.logger.debug(f"Best pickup {best_pickup}")
                facility_assigned = self._assign_best_facility(pickup=best_pickup)
                self.genetic_algorithm.logger.debug(f"Assigned to facility {facility_assigned}")
                self.s[(best_pickup, facility_assigned)] += demand
            else:
                self.genetic_algorithm.logger.debug(f"Best facility {best_facility}")
                self.x[i] = best_facility
        else:  # P2
            if best_pickup != i:
                best_facility = self._assign_best_facility(pickup=best_pickup)
                self.genetic_algorithm.logger.debug(f"Best pickup {best_pickup}")
            self.genetic_algorithm.logger.debug(f"Assgined to facility {best_facility}")
            self.w[(i, best_pickup, best_facility)] = 1
    
    def _calculate_cost_P2(self):
        cost = 0

        for (i, k, j), value in self.w.items():
            if value == 1:
                if k == i:
                    cost += self.genetic_algorithm.h[i-1] * self._get_distance_ij(i, j)
                else:
                    cost += self.genetic_algorithm.h[i-1] * self.genetic_algorithm.d_kj[k][j]
        return cost

    def _calculate_cost_P1(self):
        cost = 0
        # Coste por asignaciones directas cliente -> instalación
        for i, j in self.x.items():
            if j != 0:
                cost += self.genetic_algorithm.h[i - 1] * self._get_distance_ij(i, j)

        # Coste por asignaciones cliente -> pickup -> instalación
        for (k, j), flow in self.s.items():
            if flow > 0:
                cost += self.genetic_algorithm.d_kj[k][j] * flow

        return cost

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
        # LA POBLACIÓN ESTÁ FORMADA POR CROMOSOMAS, CADA CROMOSOMA ES UNA SOLUCIÓN
        self._create_population()
        best_fitness = float('inf')
        
        for generation in range(self.generations):
            # EVALUAMOS LA SOLUCIÓN, GUARDAMOS EL COSTE DE LA SOLUCIÓN
            fitnesses = [ind.evaluate() for ind in self.population]
            # Actualizar mejor solución (la que tenga el valor más bajo, es minimizar)
            gen_best_idx = np.argmin(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]
            # Debe ser menor pq estamos minimizando
            if gen_best_fitness < best_fitness:
                best_fitness = gen_best_fitness
                # BEST INDIVIDUAL ES UNA SOLUCIÓN COMPLETA
                self.logger.info(f"Mejor individuo {best_fitness}")
                self.best_individual = deepcopy(self.population[gen_best_idx])
            # Guardar estadísticas
            # self.best_fitness_history.append(best_fitness)
            # self.avg_fitness_history.append(np.mean(fitnesses))
            
            # Mostrar progreso
            # if generation % 50 == 0:
            #     n_open = np.sum(best_individual)
            #     print(f"Generación {generation}: Mejor fitness = {best_fitness:.2f}, "
            #           f"Facilidades abiertas = {n_open}")
            
            # Seleccionar élite
            # elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            # elite = [population[i].copy() for i in elite_indices]
            
            # # Crear nueva población
            # new_population = elite.copy()
            
            # while len(new_population) < self.population_size:
            #     
            # Selección de mejores cromosomas por sorteo, se hacen en base al valor de la función objetivo
            # Los parent son LA SOLUCIÓN COMPLETA
            parent1 = self._tournament_selection(fitnesses)
            # Deberíamos sacar al parent1 de la selección para que no fueran los dos el mismo padre?
            parent2 = self._tournament_selection(fitnesses)

            # Cruzamiento, lo haremos tanto para INSTALACIONES como para PUNTOS DE RECOGIDA
            # CASO 1. INSTALACIONES
            facilities_parent1 = parent1.y
            facilities_parent2 = parent2.y
            facilities_child1, facilities_child2 = self._crossover(parent1=facilities_parent1, parent2=facilities_parent2, n_candidates=self.n_customers, n_open=self.p)
            # CASO 2. PUNTOS DE RECOGIDA
            pickups_parent1 = parent1.nu
            pickups_parent2 = parent2.nu
            pickups_child1, pickups_child2 = self._crossover(parent1=pickups_parent1, parent2=pickups_parent2, n_candidates=self.n_pickups, n_open=self.t)
            
            # Mutación, lo haremos tanto para INSTALACIONES como para PUNTOS DE RECOGIDA
            # CASO 1. INSTALACIONES
            facilities_child1 = self._mutate(individual=facilities_child1, n_open=self.p)
            facilities_child2 = self._mutate(individual=facilities_child2, n_open=self.p)
            # CASO 2. PUNTOS DE RECOGIDA
            pickups_child1 = self._mutate(individual=pickups_child1, n_open=self.t)
            pickups_child2 = self._mutate(individual=pickups_child2, n_open=self.t)

            # NO EXTENDEMOS LA POBLACIÓN, LO QUE HACEMOS ES ESTABLECER UN CRITERIO DE SUSTITUCIÓN,
            # MIRAREMOS LOS DOS PEORES DE LA POBLACIÓN Y SE COMPARARÁN CON LOS HIJOS, LOS DOS MEJORES DE LOS 4 SERÁN LOS QUE ENTREN EN LA NUEVA POBLACIÓN
            child1 = Solution(self)
            child1.y = facilities_child1
            child1.nu = pickups_child1
            cost_child1 = child1.evaluate()

            child2 = Solution(self)
            child2.y = facilities_child2
            child2.nu = pickups_child2
            cost_child2 = child2.evaluate()

            # Obtener los índices ordenados de menor a mayor fitness
            # Índices ordenados de menor a mayor fitness
            sorted_indices = np.argsort(fitnesses)

            # Índices ordenados de mayor a menor fitness
            sorted_indices = sorted_indices[::-1]

            # Las dos peores soluciones son las primeras en el array de índices ordenados
            worst_1_idx = sorted_indices[0]
            worst_2_idx = sorted_indices[1]

            # Si quieres los valores de fitness de las peores soluciones
            worst_1_fitness = fitnesses[worst_1_idx]
            worst_2_fitness = fitnesses[worst_2_idx]
            
            # Ahora tenemos que comparar los 4 valores de coste y añadir/quitar de la pobalción los 2 mejores/peores
            selection_costs = [worst_1_fitness, worst_2_fitness, cost_child1, cost_child2]
            sorted_indices = np.argsort(selection_costs)
            sorted_indices = sorted_indices[::-1]

            worst_1 = sorted_indices[0]
            worst_2 = sorted_indices[1]
            indices_to_remove = []
            if worst_1 < 2: indices_to_remove.append(worst_1)
            if worst_2 < 2: indices_to_remove.append(worst_2)
            # Ordena los índices de mayor a menor
            indices_to_remove.sort(reverse=True)

            # Itera sobre la lista de índices y elimina
            for index in indices_to_remove:
                self.population.pop(index)

            best_1 = sorted_indices[3]
            best_2 = sorted_indices[2]

            if best_1 == 2 or best_2 == 2:
                self.population.append(child1)
            if best_1 == 3 or best_2 == 3:
                self.population.append(child2)

            # new_population.extend([child1, child2])
        
            # # Mantener tamaño de población
            # population = new_population[:self.population_size]

        solve_time = time.time() - start_time

        if self.best_individual is None:
            self.logger.warning("Error: No se encontró ninguna solución válida después de todas las iteraciones.")
            return None
        
        self._save_best_sol(solve_time)
        
        print(f"\nOptimización completada!")
        print(f"Mejor fitness: {best_fitness:.2f}")
        print(f"Facilidades seleccionadas: {self.best_individual.open_facilities}")

        
        # return {
        #     'best_solution': best_individual,
        #     'best_fitness': best_fitness,
        #     'selected_facilities': np.where(best_individual == 1)[0],
        #     'best_fitness_history': self.best_fitness_history,
        #     'avg_fitness_history': self.avg_fitness_history
        # }
    
    def _create_individual(self) -> np.ndarray:
        """Crea un individuo (solución) aleatoria"""
        # --- Inicialización de la solución ---
        cromosoma = Solution(self)
        if self.inicialization == "greedy":
            self.logger.debug("Seleccionado inicialización [GREEDY]")
            cromosoma.initialize_greedy()
        elif self.inicialization == "random":
            self.logger.debug("Seleccionado inicialización [RANDOM]")
            cromosoma.initialize_random()
        else:
            raise ValueError(f"Inicialización desconocida: {self.inicialization}")
        
        # Guardamos en una lista los cromosomas, el conjunto forma una población
        self.population.append(cromosoma)
    
    def _create_population(self) -> List[np.ndarray]:
        """Crea la población inicial"""
        for _ in range(self.population_size): self._create_individual()
    
    def _tournament_selection(self, 
                            fitnesses: List[float]) -> Solution:
        """Selección por torneo"""
        tournament = random.sample(list(zip(self.population, fitnesses)), self.tournament)
        # Como la función es de minimizar, será el mínimo de todas las funciones obejtivo
        winner = min(tournament, key=lambda x: x[1])
        # Devolvemos la solución (completa) del valor más pequeño de los que han participado en el sorteo
        return deepcopy(winner[0])
    
    # ESTO HAY QUE HACERLO PARA FACILITIES Y PARA PUNTOS DE RECOGIDA
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray, n_candidates: int, n_open: int) -> Tuple[np.ndarray, np.ndarray]:
        """Cruzamiento de dos puntos"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Dos puntos de cruzamiento
        # EN VEZ DE CANDIDATOS, CANTIDAD DE CLIENTES O PUNTOS DE REOCGIDA
        point1 = random.randint(1, n_candidates)
        # point2 = random.randint(point1 + 1, n_candidates - 1)

        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Intercambiar segmento medio
        child1[point1:] = parent2[point1:]
        child2[point1:] = parent1[point1:]
        
        # Reparar si exceden max_facilities
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
        """Repara un individuo para que respete la restricción de tener exactamente p facilidades abiertas."""

        # 1. Encontrar las facilidades abiertas y cerradas
        open = np.where(individual == 1)[0]
        closed = np.where(individual == 0)[0]

        num_open = len(open)

        if num_open > n_open:
            # 2. Si hay más de p abiertas, cerrar las sobrantes
            to_close = random.sample(list(open), num_open - n_open)
            individual[to_close] = 0
        elif num_open < n_open:
            # 3. Si hay menos de p abiertas, abrir las necesarias
            to_open = random.sample(list(closed), n_open - num_open)
            individual[to_open] = 1

        return individual
    
    def _save_best_sol(self, solve_time):
        """Guardado optimizado de la mejor solución"""
        os.makedirs(self.result_dir, exist_ok=True)
        
        filename = f"{self.instance+1}_best_solution.txt"
        filepath = os.path.join(self.result_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(f"Instancia: {self.instance+1}\n")
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