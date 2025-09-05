import os
import shutil

def copiar_archivos(base_dir="instances", destino="data"):
    # Definir qué índices j van a cada conjunto
    testing_map = {
        10: [1],
        50: [6],
        100: [15]
    }
    training_map = {
        10: [1, 16, 31, 46],
        50: [6, 21, 36, 51],
        100: [15, 30, 45, 60]
    }

    # Crear carpetas destino si no existen
    for split in ["testing_set", "training_set"]:
        for n in [10, 50, 100]:
            os.makedirs(os.path.join(destino, split, f"n_{n}"), exist_ok=True)

    # Iterar sobre p1..p10
    for i in range(1, 11):
        carpeta = os.path.join(base_dir, f"p{i}")

        # --- Testing set ---
        for n, indices in testing_map.items():
            for j in indices:
                src = os.path.join(carpeta, f"i{j}.txt")
                dst = os.path.join(destino, "testing_set", f"n_{n}", f"p{i}_i{j}.txt")
                if os.path.exists(src):
                    shutil.copy(src, dst)

        # --- Training set ---
        for n, indices in training_map.items():
            for j in indices:
                src = os.path.join(carpeta, f"i{j}.txt")
                dst = os.path.join(destino, "training_set", f"n_{n}", f"p{i}_i{j}.txt")
                if os.path.exists(src):
                    shutil.copy(src, dst)


# Ejemplo de uso:
copiar_archivos()
