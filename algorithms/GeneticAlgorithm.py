import os
import time
import traceback
import random
from typing import List, Tuple
import numpy as np

from structure.solution import Solution

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
                 tournament: int = 2,
                 mutation_rate: float = 0.05,
                 crossover_rate: float = 0.95,
                 problem: str = "P2"
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
        self.time_limit = 1000
        
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
        fitnesses = [ind.evaluate() for ind in self.population]
        best_idx = np.argmin(fitnesses)
        best_fitness = fitnesses[best_idx]

        self.best_individual = Solution(self)
        self.best_individual._clone_solution(self.population[best_idx])

        
        generation = 0

        while generation < self.generations and (time.time() - start_time) < self.time_limit:
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
            
            # Avanzar contador de generación
            generation += 1

        
        solve_time = time.time() - start_time
        self.best_individual.time = solve_time
        
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
    
    def _crossover(self, parent1, parent2, n_candidates: int, n_open: int):
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
    
    def _mutate(self, individual, n_open: int):
        """Mutación bit-flip con reparación"""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]  # Flip bit
        
        return self._repair_individual(mutated, n_open)
    
    def _repair_individual(self, individual, n_open: int):
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