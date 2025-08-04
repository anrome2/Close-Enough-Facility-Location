import os
from sklearn.cluster import KMeans

from structure.instances import readInstance

def kmeans(path):
    # Leer la instancia
    instance_dict = readInstance(path)
    p = instance_dict['p']
    coords = instance_dict['nodes']

    # KMeans para instalaciones
    kmeans = KMeans(n_clusters=p).fit(coords)
    inst_centers = kmeans.cluster_centers_
    return inst_centers
