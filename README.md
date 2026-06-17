# Práctica MPI + OpenMP: Multiplicación de matrices

Aplicación híbrida en C que multiplica dos matrices cuadradas grandes usando:

- MPI para distribuir bloques de filas entre procesos.
- OpenMP para paralelizar el cálculo dentro de cada proceso.
- `MPI_Scatterv` y `MPI_Gatherv` para soportar tamaños de matriz que no sean divisibles exactamente entre procesos.

## Estructura

```text
Practica_MPI_OpenMP/
  src/matrix_hybrid.c
  Makefile
  README.md
  docs/informe_tecnico.md
  docs/informe_tecnico.docx
  docs/informe_tecnico.pdf
  docs/plantilla_resultados.csv
  docs/guion_video.md
  tools/generar_pdf.py
```

## Requisitos

- Implementación MPI, por ejemplo OpenMPI, MPICH o Microsoft MPI.
- Compilador C compatible con OpenMP.
- `make` para usar el Makefile, aunque también se puede compilar con el comando `mpicc` indicado abajo.

## Compilacion

En Linux, WSL o un entorno HPC con MPI y OpenMP:

```bash
make
```

Comando equivalente:

```bash
mpicc -O3 -Wall -Wextra -std=c11 -fopenmp -o matrix_hybrid src/matrix_hybrid.c -lm -fopenmp
```

## Ejecucion

```bash
export OMP_NUM_THREADS=4
mpirun -np 4 ./matrix_hybrid 1024
```

Con tiempo secuencial base para calcular speedup y eficiencia:

```bash
export OMP_NUM_THREADS=4
mpirun -np 4 ./matrix_hybrid 1024 20.75
```

Con validación para tamaños pequeños:

```bash
export OMP_NUM_THREADS=2
mpirun -np 2 ./matrix_hybrid 64 0 --validate
```

## Recomendación de pruebas

Ejecutar varias combinaciones de procesos e hilos, por ejemplo:

```bash
export OMP_NUM_THREADS=1
mpirun -np 1 ./matrix_hybrid 512

export OMP_NUM_THREADS=2
mpirun -np 2 ./matrix_hybrid 512 <tiempo_secuencial>

export OMP_NUM_THREADS=4
mpirun -np 4 ./matrix_hybrid 512 <tiempo_secuencial>
```

Registrar los resultados en `docs/plantilla_resultados.csv`.
