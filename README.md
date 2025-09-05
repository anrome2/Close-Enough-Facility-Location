# CEFLP Optimization Framework

## Descripción

Framework completo para resolver el problema **Capacitated Emergency Facility Location Problem (CEFLP)** utilizando múltiples algoritmos de optimización metaheurística. El proyecto incluye implementaciones de GRASP, Algoritmos Genéticos, Búsqueda Tabú, y un orquestador para ejecutar experimentos masivos de forma paralela.

## Características Principales

- 🚀 **Algoritmos Implementados**: GRASP, Genetic Algorithm, Tabu Search, MILP
- 🔄 **Ejecución Paralela**: Soporte para multiprocesamiento con optimización de recursos
- 📊 **Análisis Automático**: Cálculo de GAPs de optimalidad y estadísticas detalladas
- 🎯 **Múltiples Modos de Ejecución**: Comités, búsqueda de hiperparámetros, ejecuciones individuales
- 📈 **Comparación con Óptimos**: Carga automática de soluciones óptimas para benchmarking
- 💾 **Resultados Estructurados**: Exportación en CSV, JSON y reportes consolidados

## Estructura del Proyecto

```
CEFLP/
├── algorithms/
│   ├── GRASP.py              # Implementación GRASP optimizada
│   ├── GeneticAlgorithm.py   # Algoritmo Genético
│   ├── TABU.py               # Búsqueda Tabú
│   └── MILP.py               # Solver MILP
├── structure/
│   ├── solution.py           # Clase Solution y evaluación
│   └── create_instances.py   # Creación de parámetros de instancia
├── data/
│   ├── training_set/         # Instancias de entrenamiento
│   ├── testing_set/          # Instancias de prueba
│   └── optimal_solutions/    # Soluciones óptimas conocidas
├── Orchestrator.py           # Motor principal de ejecución
├── main.py                   # Punto de entrada principal
└── README.md
```

## Instalación

### Prerrequisitos

```bash
Python 3.8+
```

### Dependencias

```bash
pip install pandas numpy logging concurrent-futures multiprocessing
```

### Configuración Opcional

Para usar CPLEX (recomendado para MILP):
```bash
# Instalar IBM CPLEX Studio
# Configurar variables de entorno según documentación IBM
```

## Uso Rápido

### Configuración Básica

Edita el diccionario `CONFIG` en `main.py`:

```python
CONFIG = {
    'testing': False,                    # True para instancias de test
    'n_nodos': ["n_10", "n_50", "n_100"], # Tamaños de instancia
    'algorithm': 'GRASP',               # GRASP, GENETIC, TABU, MILP
    'mode': 'committee',                # single, committee, hyperparameters
    'problem': 'P2',                    # P1 o P2
    'num_runs': 5,                      # Ejecuciones por configuración
    'num_processes': 16,                # Procesos paralelos
    # ... más parámetros específicos
}
```

### Ejecución

```bash
python main.py
```

## Modos de Ejecución

### 1. Modo Comité (`mode: 'committee'`)

Ejecuta múltiples corridas con parámetros fijos para análisis estadístico:

```python
CONFIG = {
    'mode': 'committee',
    'algorithm': 'GRASP',
    'num_runs': 10,
    'alpha': 0.3,
    'frac_neighbors': 4
}
```

**Salida**: Estadísticas agregadas (media, desviación, mejor/peor resultado, GAPs).

### 2. Búsqueda de Hiperparámetros (`mode: 'hyperparameters'`)

Explora diferentes combinaciones de parámetros:

```python
CONFIG = {
    'mode': 'hyperparameters',
    'algorithm': 'GENETIC',
    'type_search': 'grid',  # grid, random
    'num_runs': 8
}
```

**Salida**: Ranking de mejores configuraciones, análisis comparativo.

### 3. Ejecución Individual (`mode: 'single'`)

Una sola ejecución por instancia con parámetros específicos:

```python
CONFIG = {
    'mode': 'single',
    'algorithm': 'TABU',
    'tabu_tenure': 0.35,
    'time_limit': 120
}
```

## Algoritmos Disponibles

### GRASP (Greedy Randomized Adaptive Search Procedure)

```python
# Parámetros principales
{
    'alpha': 0.3,           # Factor de aleatoriedad (0-1)
    'frac_neighbors': 4,    # Fracción de vecindario
    'max_iter': 50          # Iteraciones máximas
}
```

**Características**:
- Construcción greedy randomizada
- Búsqueda local con listas pivotales
- Criterio de parada adaptativo

### Algoritmo Genético

```python
# Parámetros principales
{
    'mutation_rate': 0.05,     # Tasa de mutación
    'tournament': 5,           # Tamaño del torneo
    'inicializacion': 'random' # random, greedy
}
```

**Características**:
- Selección por torneo
- Mutación adaptativa
- Múltiples estrategias de inicialización

### Búsqueda Tabú

```python
# Parámetros principales
{
    'tabu_tenure': 0.25,       # Duración tabú (fracción del problema)
    'time_limit': 120,         # Límite de tiempo en segundos
    'inicializacion': 'kmeans' # random, greedy, kmeans
}
```

**Características**:
- Lista tabú adaptativa
- Criterios de aspiración
- Inicialización inteligente con K-means

## Análisis de Resultados

### Métricas Automáticas

- **GAP de Optimalidad**: `GAP = |costo_encontrado - costo_óptimo| / costo_óptimo × 100`
- **Estadísticas Descriptivas**: Media, desviación estándar, percentiles
- **Análisis de Convergencia**: Tiempo hasta mejor solución
- **Tasa de Éxito**: Porcentaje de soluciones óptimas encontradas

### Estructura de Salida

```
output/P2/GRASP/committee/20241201_143022/
├── GRASP_committee_summary.csv          # Resumen por instancia
├── GRASP_committee_detailed.json        # Datos detallados
├── GRASP_all_results.csv               # Todas las ejecuciones
├── GRASP_global_stats_by_nnodes.csv    # Estadísticas por tamaño
└── REPORTE_FINAL.txt                   # Reporte consolidado
```

### Ejemplo de Análisis

```python
# Cargar resultados
import pandas as pd
results = pd.read_csv('GRASP_committee_summary.csv')

# Mejores instancias por GAP
best_gaps = results.nsmallest(10, 'best_gap')
print(best_gaps[['instance_name', 'best_cost', 'best_gap', 'optimal_solutions']])

# Análisis por tamaño
size_analysis = results.groupby('n_nodes').agg({
    'best_gap': ['mean', 'std'],
    'best_cost': 'mean',
    'optimal_solutions': 'sum'
})
```

## Optimización y Rendimiento

### Paralelización Inteligente

- **Nivel de Instancia**: Múltiples instancias en paralelo
- **Nivel de Algoritmo**: Ejecuciones paralelas por configuración
- **Gestión de Memoria**: Limpieza automática de directorios temporales
- **Balanceado de Carga**: Distribución adaptativa según recursos disponibles

### Mejoras Implementadas

1. **GRASP Optimizado**:
   - Listas pivotales para búsqueda local dirigida
   - Pre-computación de estructuras de datos
   - Criterios de parada temprana

2. **Evaluación Eficiente**:
   - Evaluación incremental de soluciones
   - Cache de distancias pre-computadas
   - Estructuras de datos optimizadas

3. **Gestión de Resultados**:
   - Compresión automática de logs
   - Exportación en múltiples formatos
   - Resúmenes ejecutivos automáticos

## Configuración Avanzada

### Parámetros por Algoritmo

#### GRASP Avanzado
```python
grasp_config = {
    'alpha': 0.3,                    # Aleatorización construcción
    'frac_neighbors': 4,             # Intensidad búsqueda local
    'max_iter': 50,                  # Límite iteraciones
    'early_stop_threshold': 10,      # Parada temprana
    'pivotal_list_size': 0.5         # Fracción lista pivotal
}
```

#### Algoritmo Genético Avanzado
```python
genetic_config = {
    'population_size': 50,           # Tamaño población
    'mutation_rate': 0.05,           # Tasa mutación
    'crossover_rate': 0.95,          # Tasa cruce
    'tournament': 5,                 # Tamaño torneo
    'elitism_rate': 0.1,             # Fracción élite
    'diversity_threshold': 0.8       # Umbral diversidad
}
```

#### Búsqueda Tabú Avanzada
```python
tabu_config = {
    'tabu_tenure': 0.25,             # Duración tabú
    'aspiration_criterion': True,     # Criterio aspiración
    'intensification_freq': 20,       # Frecuencia intensificación
    'diversification_freq': 50,       # Frecuencia diversificación
    'memory_type': 'frequency'        # Tipo memoria: frequency/recency
}
```

### Configuración de Experimentos Masivos

```python
# Experimento comparativo completo
EXPERIMENT_CONFIG = {
    'algorithms': ['GRASP', 'GENETIC', 'TABU'],
    'modes': ['committee', 'hyperparameters'],
    'instance_sizes': ['n_10', 'n_50', 'n_100'],
    'replications': 20,
    'max_parallel_jobs': 32,
    'timeout_per_instance': 300,
    'save_intermediate_results': True
}
```

## Solución de Problemas Comunes

### Errores de Memoria

```python
# Reducir uso de memoria
CONFIG.update({
    'num_processes': 4,              # Reducir procesos
    'batch_size': 50,                # Procesar en lotes
    'cleanup_temp_files': True       # Limpiar archivos temporales
})
```

### Timeouts en Instancias Grandes

```python
# Ajustar límites de tiempo
CONFIG.update({
    'time_limit': 600,               # 10 minutos por instancia
    'max_iter': 100,                 # Más iteraciones
    'early_stop_patience': 20        # Paciencia para parada temprana
})
```

### Problemas de Convergencia

```python
# Mejorar exploración
grasp_config.update({
    'alpha': 0.5,                    # Mayor aleatorización
    'frac_neighbors': 2,             # Búsqueda más intensiva
    'restart_frequency': 25          # Reiniciar periódicamente
})
```

## Extensión del Framework

### Añadir Nuevo Algoritmo

1. **Crear clase del algoritmo**:
```python
# algorithms/NuevoAlgoritmo.py
class NuevoAlgoritmo:
    def __init__(self, params, **kwargs):
        self.params = params
        # Inicialización específica
    
    def run(self):
        # Implementación del algoritmo
        pass
```

2. **Registrar en Orchestrator**:
```python
# En run_single_algorithm_execution()
elif algorithm_type == 'NUEVO':
    return run_single_nuevo_execution(args)
```

3. **Configurar generación de parámetros**:
```python
def generate_nuevo_param_combinations(search_type="grid"):
    # Definir espacio de búsqueda
    return combinations
```

### Métricas Personalizadas

```python
# En analyze_results()
def custom_analysis(costs, times, optimal_cost=None):
    analysis = standard_analysis(costs, times, optimal_cost)
    
    # Añadir métricas personalizadas
    analysis.update({
        'convergence_rate': calculate_convergence(costs),
        'stability_index': calculate_stability(costs),
        'efficiency_ratio': calculate_efficiency(costs, times)
    })
    
    return analysis
```

## Contribución

### Guías de Desarrollo

1. **Estilo de Código**: Seguir PEP 8
2. **Testing**: Incluir tests unitarios para nuevas funcionalidades
3. **Documentación**: Documentar funciones con docstrings
4. **Logging**: Usar el sistema de logging existente

### Estructura de Commits

```
feat: Añadir nuevo algoritmo XYZ
fix: Corregir cálculo de GAP en instancias P2
docs: Actualizar README con ejemplos avanzados
perf: Optimizar evaluación de soluciones en GRASP
```

## Referencias

- Drezner, Z., & Hamacher, H. W. (2001). *Facility location: applications and theory*
- Resende, M. G., & Ribeiro, C. C. (2016). *Optimization by GRASP*
- Glover, F., & Laguna, M. (1997). *Tabu Search*

---

**Última actualización**: Septiembre 2025
**Compatibilidad**: Python 3.11.2+
