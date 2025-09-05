import os
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_grasp_results():
    """
    Analiza los resultados del algoritmo GRASP agrupados por número de nodos.
    """
    
    # Definir los grupos y sus correspondientes valores de 'i'
    groups = {
        '10_nodos': ['i1', 'i16', 'i31', 'i46'],
        '50_nodos': ['i6', 'i21', 'i36', 'i51'], 
        '100_nodos': ['i15', 'i30', 'i45', 'i60']
    }
    
    # Ruta de la carpeta con los resultados
    results_folder = Path('results/GRASP_fixed_params_results')
    
    # Lista para almacenar las estadísticas finales
    final_stats = []
    
    # Verificar que la carpeta existe
    if not results_folder.exists():
        print(f"Error: La carpeta {results_folder} no existe")
        return
    
    # Procesar cada grupo
    for group_name, i_values in groups.items():
        print(f"\nProcesando grupo: {group_name}")
        
        group_data = {
            'costes': [],
            'duraciones': [],
            'gaps': [],
            'factibles_pct': [],
            'optimas_pct': []
        }
        
        files_found = 0
        
        # Buscar archivos para este grupo
        for i_value in i_values:
            # Buscar archivos con el patrón GRASP_p*_{i_value}_results.csv
            pattern = f"GRASP_p*_{i_value}_results.csv"
            matching_files = list(results_folder.glob(pattern))
            
            for file_path in matching_files:
                try:
                    # Leer el archivo CSV
                    df = pd.read_csv(file_path)
                    files_found += 1
                    
                    print(f"  Procesando: {file_path.name}")
                    
                    # Verificar que las columnas necesarias existen
                    required_cols = ['cost', 'time', 'success', 'gap', 'is_optimal']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        print(f"    Advertencia: Columnas faltantes en {file_path.name}: {missing_cols}")
                        continue
                    
                    # Calcular estadísticas para esta instancia
                    # Solo considerar filas con datos válidos
                    valid_data = df.dropna()
                    
                    if len(valid_data) == 0:
                        print(f"    Advertencia: No hay datos válidos en {file_path.name}")
                        continue
                    
                    # Coste medio de esta instancia
                    coste_medio = valid_data['cost'].mean()
                    group_data['costes'].append(coste_medio)
                    
                    # Duración media de esta instancia
                    duracion_media = valid_data['time'].mean()
                    group_data['duraciones'].append(duracion_media)
                    
                    # Gap medio de esta instancia (solo de ejecuciones factibles)
                    factibles = valid_data[valid_data['success'] == True]
                    if len(factibles) > 0:
                        gap_medio = factibles['gap'].mean()
                        group_data['gaps'].append(gap_medio)
                    
                    # Porcentaje de ejecuciones factibles en esta instancia
                    pct_factibles = (valid_data['success'].sum() / len(valid_data)) * 100
                    group_data['factibles_pct'].append(pct_factibles)
                    
                    # Porcentaje de ejecuciones óptimas en esta instancia
                    pct_optimas = (valid_data['is_optimal'].sum() / len(valid_data)) * 100
                    group_data['optimas_pct'].append(pct_optimas)
                    
                except Exception as e:
                    print(f"    Error procesando {file_path.name}: {str(e)}")
                    continue
        
        if files_found == 0:
            print(f"  Advertencia: No se encontraron archivos para el grupo {group_name}")
            continue
        
        # Calcular estadísticas globales del grupo
        stats = {
            'grupo': group_name,
            'num_instancias': len(group_data['costes']),
            'coste_medio': np.mean(group_data['costes']) if group_data['costes'] else np.nan,
            'duracion_media': np.mean(group_data['duraciones']) if group_data['duraciones'] else np.nan,
            'gap_medio': np.mean(group_data['gaps']) if group_data['gaps'] else np.nan,
            'porcentaje_factibles': np.mean(group_data['factibles_pct']) if group_data['factibles_pct'] else np.nan,
            'porcentaje_optimas': np.mean(group_data['optimas_pct']) if group_data['optimas_pct'] else np.nan
        }
        
        final_stats.append(stats)
        
        # Mostrar resumen del grupo
        print(f"  Instancias procesadas: {stats['num_instancias']}")
        print(f"  Coste medio: {stats['coste_medio']:.2f}")
        print(f"  Duración media: {stats['duracion_media']:.2f}")
        print(f"  Gap medio: {stats['gap_medio']:.2f}%")
        print(f"  % Factibles: {stats['porcentaje_factibles']:.1f}%")
        print(f"  % Óptimas: {stats['porcentaje_optimas']:.1f}%")
    
    # Crear DataFrame con los resultados
    results_df = pd.DataFrame(final_stats)
    
    # Guardar resultados
    output_file = 'estadisticas_grasp_por_grupos.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Estadísticas guardadas en: {output_file}")
    
    # Mostrar tabla resumen
    print("\n" + "="*80)
    print("RESUMEN DE ESTADÍSTICAS POR GRUPO")
    print("="*80)
    print(results_df.to_string(index=False, float_format='%.2f'))
    
    return results_df

def main():
    """Función principal"""
    print("Analizador de resultados GRASP")
    print("-" * 50)
    
    try:
        results = analyze_grasp_results()
        
        if results is not None and len(results) > 0:
            print(f"\n✓ Análisis completado exitosamente")
            print(f"✓ Se procesaron {len(results)} grupos")
        else:
            print("\n⚠ No se pudieron procesar datos")
            
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {str(e)}")

if __name__ == "__main__":
    main()