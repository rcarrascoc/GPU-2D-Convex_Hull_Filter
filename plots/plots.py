import pandas as pd

# File path
file_path = '../data/normal_dist.dat'  # Asegúrate de usar la ruta correcta a tu archivo

# Inicializar listas para cada columna
numero = []
cgal_ch_2 = []
cgal_graham_andrew = []
cpu_manhattan = []
cpu_euclidean = []
gpu_scan = []
cub_flagged = []
thrust_scan = []
thrust_copy = []
omp_manhattan = []
omp_euclidean = []

# Leer el archivo y extraer los datos requeridos
with open(file_path, "r") as file:
    lines = file.readlines()

for line in lines:
    data = line.strip().split()

    if len(data) >= 47:  # Asegurarse de que la línea tiene suficientes puntos de datos
        numero.append(int(data[0]))
        cgal_ch_2.append(float(data[1]))
        cgal_graham_andrew.append(float(data[6]))
        cpu_manhattan.append(float(data[11]))
        cpu_euclidean.append(float(data[16]))
        gpu_scan.append(float(data[21]))
        cub_flagged.append(float(data[26]))
        thrust_scan.append(float(data[31]))
        thrust_copy.append(float(data[36]))
        omp_manhattan.append(float(data[41]))
        omp_euclidean.append(float(data[46]))

# Crear un diccionario para convertirlo en un DataFrame
data_dict = {
    "points": numero,
    "cgal_ch_2": cgal_ch_2,
    "cgal_graham_andrew": cgal_graham_andrew,
    "cpu_manhattan": cpu_manhattan,
    "cpu_euclidean": cpu_euclidean,
    "gpu_scan": gpu_scan,
    "cub_flagged": cub_flagged,
    "thrust_scan": thrust_scan,
    "thrust_copy": thrust_copy,
    "omp_manhattan": omp_manhattan,
    "omp_euclidean": omp_euclidean
}

# Converting the updated dictionary into a DataFrame
selected_data_df = pd.DataFrame(data_dict)

# Displaying the first few rows of the updated DataFrame

import matplotlib.pyplot as plt

# Plotting the data
plt.figure(figsize=(12, 8))

# Plotting each metric
plt.plot(selected_data_df['points'], selected_data_df['cgal_ch_2'], label='cgal_ch_2')
#plt.plot(selected_data_df['points'], selected_data_df['cgal_graham_andrew'], label='cgal_graham_andrew')
plt.plot(selected_data_df['points'], selected_data_df['cpu_manhattan'], label='CPU Manhattan')
plt.plot(selected_data_df['points'], selected_data_df['cpu_euclidean'], label='CPU Euclidean')
plt.plot(selected_data_df['points'], selected_data_df['gpu_scan'], label='GPU Scan')
plt.plot(selected_data_df['points'], selected_data_df['cub_flagged'], label='CUB Flagged')
plt.plot(selected_data_df['points'], selected_data_df['thrust_scan'], label='Thrust Scan')
plt.plot(selected_data_df['points'], selected_data_df['thrust_copy'], label='Thrust Copy')
plt.plot(selected_data_df['points'], selected_data_df['omp_manhattan'], label='OMP Manhattan')
plt.plot(selected_data_df['points'], selected_data_df['omp_euclidean'], label='OMP Euclidean')

# Adding titles and labels
plt.title('Performance Metrics across Different Points')
plt.xlabel('Points')
plt.ylabel('Performance Metrics')
plt.legend()

# Show the plot
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Asumiendo que 'selected_data_df' ya tiene los datos necesarios
# Calculando el speedup para cada métrica en comparación con 'cpu_euclidean'
speedup_data_df = selected_data_df.copy()
for column in speedup_data_df.columns:
    if column != 'points' and column != 'cgal_ch_2':
        speedup_data_df[column] = speedup_data_df['cgal_ch_2'] / speedup_data_df[column]

# Plotting the speedup data
plt.figure(figsize=(12, 8))

# Plotting each metric's speedup
for column in speedup_data_df.columns:
    if column != 'points' and column != 'cgal_ch_2':
        plt.plot(speedup_data_df['points'], speedup_data_df[column], label=f'{column}')

# Adding titles and labels
plt.title('Speedup Compared to cgal_ch_2 in a Normal Distribution')
plt.xlabel('Points')
plt.ylabel('Speedup Factor')
plt.legend()

# export to pdf
plt.savefig('normal_dist.pdf')

# Show the plot
plt.show()
