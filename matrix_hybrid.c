#include <mpi.h>
#include <omp.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die_if(int condition, const char *message, int rank)
{
    if (condition) {
        if (rank == 0) {
            fprintf(stderr, "Error: %s\n", message);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

static double *allocate_matrix(int rows, int cols, int rank)
{
    size_t count = (size_t)rows * (size_t)cols;
    double *matrix = (double *)malloc(count * sizeof(double));
    die_if(matrix == NULL, "no se pudo reservar memoria para la matriz", rank);
    return matrix;
}

static void initialize_matrix(double *matrix, int rows, int cols, int seed)
{
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int value = (i * 31 + j * 17 + seed) % 97;
            matrix[i * cols + j] = (double)value / 10.0;
        }
    }
}

static void build_counts(int n, int size, int *rows, int *counts, int *displs)
{
    int base = n / size;
    int remainder = n % size;
    int offset = 0;

    for (int p = 0; p < size; ++p) {
        rows[p] = base + (p < remainder ? 1 : 0);
        counts[p] = rows[p] * n;
        displs[p] = offset;
        offset += counts[p];
    }
}

static void multiply_block(const double *local_a,
                           const double *b,
                           double *local_c,
                           int local_rows,
                           int n)
{
    int i;
#pragma omp parallel for schedule(static)
    for (i = 0; i < local_rows; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += local_a[i * n + k] * b[k * n + j];
            }
            local_c[i * n + j] = sum;
        }
    }
}

static int validate_result(const double *a, const double *b, const double *c, int n)
{
    const double tolerance = 1e-8;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double expected = 0.0;
            for (int k = 0; k < n; ++k) {
                expected += a[i * n + k] * b[k * n + j];
            }
            if (fabs(expected - c[i * n + j]) > tolerance) {
                fprintf(stderr,
                        "Fallo de validacion en C[%d][%d]: esperado %.6f, obtenido %.6f\n",
                        i,
                        j,
                        expected,
                        c[i * n + j]);
                return 0;
            }
        }
    }

    return 1;
}

static void print_usage(const char *program)
{
    printf("Uso: %s <tamano_matriz> [tiempo_secuencial] [--validate]\n", program);
    printf("Ejemplo: mpirun -np 4 %s 1024 12.50\n", program);
    printf("Si se pasa tiempo_secuencial, se calculan speedup y eficiencia.\n");
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    double sequential_time = 0.0;
    int validate = 0;

    for (int arg = 2; arg < argc; ++arg) {
        if (strcmp(argv[arg], "--validate") == 0) {
            validate = 1;
        } else {
            sequential_time = atof(argv[arg]);
        }
    }

    die_if(n <= 0, "el tamano de la matriz debe ser positivo", rank);

    int *rows = (int *)malloc((size_t)size * sizeof(int));
    int *counts = (int *)malloc((size_t)size * sizeof(int));
    int *displs = (int *)malloc((size_t)size * sizeof(int));
    die_if(rows == NULL || counts == NULL || displs == NULL,
           "no se pudo reservar memoria para metadatos de distribucion",
           rank);

    /* Se reparte A por filas. Scatterv/Gatherv soportan N no divisible por size. */
    build_counts(n, size, rows, counts, displs);

    double *a = NULL;
    double *c = NULL;
    double *b = allocate_matrix(n, n, rank);
    double *local_a = allocate_matrix(rows[rank], n, rank);
    double *local_c = allocate_matrix(rows[rank], n, rank);

    if (rank == 0) {
        a = allocate_matrix(n, n, rank);
        c = allocate_matrix(n, n, rank);
        initialize_matrix(a, n, n, 11);
        initialize_matrix(b, n, n, 29);
    }

    /* Cada proceso necesita B completa para calcular sus filas de C. */
    MPI_Bcast(b, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatterv(a,
                 counts,
                 displs,
                 MPI_DOUBLE,
                 local_a,
                 counts[rank],
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    /* OpenMP paraleliza el calculo local dentro de cada proceso MPI. */
    multiply_block(local_a, b, local_c, rows[rank], n);

    MPI_Gatherv(local_c,
                counts[rank],
                MPI_DOUBLE,
                c,
                counts,
                displs,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start;

    if (rank == 0) {
        int threads = omp_get_max_threads();
        int total_workers = size * threads;
        double speedup = sequential_time > 0.0 ? sequential_time / elapsed : 0.0;
        double efficiency = speedup > 0.0 ? speedup / (double)total_workers : 0.0;

        printf("Tamano matriz: %d x %d\n", n, n);
        printf("Procesos MPI: %d\n", size);
        printf("Hilos OpenMP por proceso: %d\n", threads);
        printf("Tiempo paralelo: %.6f segundos\n", elapsed);

        if (sequential_time > 0.0) {
            printf("Tiempo secuencial base: %.6f segundos\n", sequential_time);
            printf("Speedup: %.4f\n", speedup);
            printf("Eficiencia: %.4f (%.2f%%)\n", efficiency, efficiency * 100.0);
        }

        if (validate) {
            printf("Validacion: %s\n", validate_result(a, b, c, n) ? "correcta" : "incorrecta");
        }
    }

    free(rows);
    free(counts);
    free(displs);
    free(a);
    free(b);
    free(c);
    free(local_a);
    free(local_c);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
