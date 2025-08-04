# Datos de instancias para el problema de p-mediana

Estos archivos contienen datos para instancias de un problema de p-mediana, organizados en distintos problemas (20) con varios clientes y demandas.

En la carpeta data se han guardado los diferentes archivos .txt, uno diferente para cada problema de la siguiente forma:
/data
├── problem1.txt
├── problem2.txt
└── ...

El archivo pmedcap1.txt sigue este formato:
- Primera fila: Número total de problemas (`M`).
- Segunda fila: Cada problema identificado con `m`, columna 1; columna 2 ignorable.
- Tercera fila: Número de clientes (`N`) para el problema.
- Siguientes filas: Información de cada cliente con los siguientes datos:
  - Columna 1: ID del cliente
  - Columnas 2 y 3: Coordenadas X e Y del cliente
  - Columna 4: Demanda del cliente
  
 Y de la lectura del archivo pmedcap1.txt hemos extraido los datos para cada problemX.txt, siguiendo estos archivos la siguiente estructura:
- Columna 1: ID del cliente
- Columnas 2 y 3: Coordenadas X e Y del cliente
- Columna 4: Demanda del cliente

Para poder usar estos archivos en Python se leerán de la siguiente forma:

``
import pandas as pd

# Leer archivo de ejemplo
df = pd.read_csv('data/problem1.txt', sep=' ')

``
