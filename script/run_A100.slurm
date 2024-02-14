#!/bin/bash
#SBATCH --job-name=2D     # Nombre del trabajo
#SBATCH --output=dist_test_%j.out       # Nombre del archivo de salida
#SBATCH --error=dist_test_%j.err        # Nombre del archivo de error
#SBATCH --ntasks=1                      # Numero de tareas
#SBATCH --gres=gpu:A100:1               # Solicitar un GPU A100
#SBATCH --chdir=/home/rcarrasco/project/GPU-2D-Convex_Hull_Filter/script

# Configuración inicial
STARTN=$1  # Tamaño inicial
DN=$2      # Incremento del tamaño
ENDN=$3    # Tamaño final
SAMPLES=$4 # Número de muestras
OUTFILE=$5 # Archivo de salida
BINARY=$6  # Nombre del binario a ejecutar
SHAPE=$7   # Parámetro de forma
PROB=$8    # Probabilidad
PATH=$9    # Ruta (no se usa en sbatch, considerar remover si no es necesario)

echo "Iniciando ejecución del script para distintos tamaños de entrada y algoritmos"

# Preparar el ambiente, si es necesario
# module load cuda/12.1.0

# STARTN=$1; DN=$2; ENDN=$3; SAMPLES=$4; OUTFILE=$5; BINARY=$6; SHAPE=$7; PROB=$8; PATH=$9;


# Corrección: Uso de bucle for en Bash en lugar de `seq`
for ((N=STARTN; N<=ENDN; N+=DN));
do
    echo -n "${N}  " >> "${PATH}/${OUTFILE}"
    for ALG in 6 7 0 1 2 3 4 5 8 9;
    do
        for ((k=1; k<=SAMPLES; k++));
        do
            echo  "./${BINARY} ${N} ${ALG} $SHAPE  $PROB $PATH"
            value=`./${BINARY} ${N} ${ALG} $SHAPE  $PROB $PATH`
        done
    done
    echo " "
    echo " " >> "${PATH}/${OUTFILE}"
done
echo " " >> "${PATH}/${OUTFILE}"
echo " "

echo "Finalizando ejecución del script"

# Finalizar el trabajo
exit 0
