from main import main

# Config base
BASE_CONFIG = {
    'testing': True,
    'n_nodos': [],  # Se rellenará en el loop
    'save_results': True,
    'problem': 'P2',
    'algorithm': '',  # Se define en el loop
    'optimization_solver': 'CPLEX',
    'inicialization': 'random',
    'mode': 'hyperparameters',
    'type_search': 'grid',
    'alpha': 0.65,
    'frac_neighbors': 3,
    'tabu_tenure': 0.35,
    'tournament': 2,
    'gamma_f': 0.5,
    'gamma_q': 0.5,
    'time_limit': 50,
    'max_iter': 50,
    'num_runs': 2,
    'num_processes': 16,  # Se ajusta según nodos
}

def run_experiments():
    # Orden de nodos
    nodos = ["n_100"]

    # Orden de algoritmos
    algoritmos = ["GRASP", "GENETIC"]

    for algoritmo in algoritmos:
        for nodo in nodos:
            # Copia de la config base
            config = BASE_CONFIG.copy()
            
            # Ajustar algoritmo y nodos
            config['algorithm'] = algoritmo
            config['n_nodos'] = [nodo]

            # Ajustar procesos según nodo
            # if nodo in ["n_10"]:
            #     config['num_processes'] = 16
            # else:  # n_100
            #     config['num_processes'] = 8

            print(f"\n=== Ejecutando {algoritmo} con {nodo} usando {config['num_processes']} hilos ===")
            main(config)

if __name__ == "__main__":
    run_experiments()
