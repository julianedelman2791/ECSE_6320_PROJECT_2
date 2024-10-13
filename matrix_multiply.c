#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <arm_neon.h>

#define MAX_THREADS 8
#define BLOCK_SIZE 64

int NUM_THREADS = 1;
int USE_SIMD = 0;
int USE_CACHE_OPT = 0;
int USE_MULTITHREADING = 0;

typedef struct {
    int start_row;
    int end_row;
    int size;
    float *A;
    float *B_T;
    float *C;
} ThreadData;

// Function prototypes
void *matrix_multiply(void *arg);
void generate_dense_matrix(float *matrix, int size);
void generate_sparse_matrix(float *matrix, int size, float sparsity);
void print_matrix(float *matrix, int size);
void transpose_matrix(float *B, float *B_T, int size);

// Utility function to check for memory allocation errors
void check_allocation(void *ptr, const char *msg) {
    if (ptr == NULL) {
        fprintf(stderr, "Error: %s\n", msg);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    int size = 1000;
    float sparsity_A = 1.0f;
    float sparsity_B = 1.0f;

    // Argument parsing
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--size") == 0) {
            if (i + 1 < argc) {
                size = atoi(argv[++i]);
                if (size <= 0) {
                    fprintf(stderr, "Error: Matrix size must be positive.\n");
                    return EXIT_FAILURE;
                }
            } else {
                fprintf(stderr, "Error: --size flag requires an integer value.\n");
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "--threads") == 0) {
            if (i + 1 < argc) {
                NUM_THREADS = atoi(argv[++i]);
                if (NUM_THREADS <= 0 || NUM_THREADS > MAX_THREADS) {
                    fprintf(stderr, "Error: Number of threads must be between 1 and %d.\n", MAX_THREADS);
                    return EXIT_FAILURE;
                }
                USE_MULTITHREADING = 1;
            } else {
                fprintf(stderr, "Error: --threads flag requires an integer value.\n");
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "--simd") == 0) {
            #if defined(__ARM_NEON)
                USE_SIMD = 1;
            #else
                fprintf(stderr, "Warning: SIMD optimizations not supported on this architecture.\n");
            #endif
        }
        else if (strcmp(argv[i], "--cache-opt") == 0) {
            USE_CACHE_OPT = 1;
        }
        else if (strcmp(argv[i], "--sparsity-A") == 0) {
            if (i + 1 < argc) {
                sparsity_A = atof(argv[++i]);
                if (sparsity_A < 0.0f || sparsity_A > 1.0f) {
                    fprintf(stderr, "Error: Sparsity-A must be between 0 and 1.\n");
                    return EXIT_FAILURE;
                }
            } else {
                fprintf(stderr, "Error: --sparsity-A flag requires a float value.\n");
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "--sparsity-B") == 0) {
            if (i + 1 < argc) {
                sparsity_B = atof(argv[++i]);
                if (sparsity_B < 0.0f || sparsity_B > 1.0f) {
                    fprintf(stderr, "Error: Sparsity-B must be between 0 and 1.\n");
                    return EXIT_FAILURE;
                }
            } else {
                fprintf(stderr, "Error: --sparsity-B flag requires a float value.\n");
                return EXIT_FAILURE;
            }
        }
        else {
            fprintf(stderr, "Error: Unknown argument '%s'\n", argv[i]);
            fprintf(stderr, "Usage: %s --size [matrix_size] --sparsity-A [sparsity] --sparsity-B [sparsity] [--threads N] [--simd] [--cache-opt]\n", argv[0]);
            return EXIT_FAILURE;
        }
    }

    // Allocate matrices as contiguous 1D arrays
    float *A = (float *)malloc(size * size * sizeof(float));
    float *B = (float *)malloc(size * size * sizeof(float));
    float *C = (float *)malloc(size * size * sizeof(float));
    float *B_T = (float *)malloc(size * size * sizeof(float));

    check_allocation(A, "Failed to allocate memory for matrix A.");
    check_allocation(B, "Failed to allocate memory for matrix B.");
    check_allocation(C, "Failed to allocate memory for matrix C.");
    check_allocation(B_T, "Failed to allocate memory for transposed matrix B.");

    // Initialize C to zero
    memset(C, 0, size * size * sizeof(float));

    printf("Matrix allocation and initialization completed.\n");
    fflush(stdout);

    // Generate matrices
    if (sparsity_A < 1.0f) {
        printf("Generating sparse matrix A with sparsity %.2f%%...\n", sparsity_A * 100);
        fflush(stdout);
        generate_sparse_matrix(A, size, sparsity_A);
    }
    else {
        printf("Generating dense matrix A...\n");
        fflush(stdout);
        generate_dense_matrix(A, size);
    }

    if (sparsity_B < 1.0f) {
        printf("Generating sparse matrix B with sparsity %.2f%%...\n", sparsity_B * 100);
        fflush(stdout);
        generate_sparse_matrix(B, size, sparsity_B);
    }
    else {
        printf("Generating dense matrix B...\n");
        fflush(stdout);
        generate_dense_matrix(B, size);
    }

    // Transpose matrix B to B_T for better cache access
    printf("Transposing matrix B...\n");
    fflush(stdout);
    transpose_matrix(B, B_T, size);
    printf("Matrix transposition completed.\n");
    fflush(stdout);

    // Start time measurement using clock_gettime for high-resolution timing
    struct timespec start_time, end_time;
    if (clock_gettime(CLOCK_MONOTONIC, &start_time) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }

    if (USE_MULTITHREADING) {
        pthread_t threads[NUM_THREADS];
        ThreadData thread_data[NUM_THREADS];
        int rows_per_thread = size / NUM_THREADS;

        for (int i = 0; i < NUM_THREADS; i++) {
            thread_data[i].start_row = i * rows_per_thread;
            thread_data[i].end_row = (i == NUM_THREADS - 1) ? size : (i + 1) * rows_per_thread;
            thread_data[i].size = size;
            thread_data[i].A = A;
            thread_data[i].B_T = B_T;
            thread_data[i].C = C;

            int rc = pthread_create(&threads[i], NULL, matrix_multiply, (void *)&thread_data[i]);
            if (rc) {
                fprintf(stderr, "Error: Unable to create thread %d\n", i);
                exit(EXIT_FAILURE);
            }
        }

        // Join threads
        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(threads[i], NULL);
        }
    } else {
        // Single-threaded execution
        ThreadData data;
        data.start_row = 0;
        data.end_row = size;
        data.size = size;
        data.A = A;
        data.B_T = B_T;
        data.C = C;

        matrix_multiply((void *)&data);
    }

    // End time measurement
    if (clock_gettime(CLOCK_MONOTONIC, &end_time) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }

    // Calculate elapsed time in seconds
    double time_spent = (end_time.tv_sec - start_time.tv_sec) +
                        (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    printf("Time taken: %f seconds\n", time_spent);
    fflush(stdout);

    // Free memory
    free(A);
    free(B);
    free(B_T);
    free(C);

    return 0;
}

void generate_dense_matrix(float *matrix, int size) {
    for (int i = 0; i < size * size; i++)
        matrix[i] = (float)(rand() % 10 + 1);
}

void generate_sparse_matrix(float *matrix, int size, float sparsity) {
    int total_elements = size * size;
    int non_zero_elements = (int)(total_elements * sparsity);

    // Initialize all elements to zero
    memset(matrix, 0, size * size * sizeof(float));

    for (int n = 0; n < non_zero_elements; n++) {
        int idx = rand() % total_elements;
        if (matrix[idx] == 0.0f)
            matrix[idx] = (float)(rand() % 10 + 1);
        else
            n--;
    }
}

void transpose_matrix(float *B, float *B_T, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            B_T[j * size + i] = B[i * size + j];
        }
    }
}

void *matrix_multiply(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    int size = data->size;
    float *A = data->A;
    float *B_T = data->B_T;
    float *C = data->C;
    int start_row = data->start_row;
    int end_row = data->end_row;

    if (USE_SIMD) {
        // SIMD-optimized multiplication using NEON intrinsics
        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < size; j++) {
                float sum = 0.0f;
                int k;
                for (k = 0; k <= size - 4; k += 4) {
                    float32x4_t vec_a = vld1q_f32(&A[i * size + k]);
                    float32x4_t vec_b = vld1q_f32(&B_T[j * size + k]);
                    float32x4_t vec_mul = vmulq_f32(vec_a, vec_b);
                    float32x4_t vec_sum = vpaddq_f32(vec_mul, vec_mul);
                    vec_sum = vpaddq_f32(vec_sum, vec_sum);
                    sum += vgetq_lane_f32(vec_sum, 0);
                }
                for (; k < size; k++) {
                    sum += A[i * size + k] * B_T[j * size + k];
                }
                C[i * size + j] = sum;
            }
        }
    } else if (USE_CACHE_OPT) {
        // Cache-optimized multiplication using loop tiling/blocking
        int block_size = BLOCK_SIZE;
        for (int i = start_row; i < end_row; i += block_size) {
            int i_max = (i + block_size > end_row) ? end_row : (i + block_size);
            for (int j = 0; j < size; j += block_size) {
                for (int k = 0; k < size; k += block_size) {
                    int i_end = (i + block_size > i_max) ? i_max : (i + block_size);
                    int j_end = (j + block_size > size) ? size : (j + block_size);
                    int k_end = (k + block_size > size) ? size : (k + block_size);
                    for (int ii = i; ii < i_end; ii++) {
                        for (int jj = j; jj < j_end; jj++) {
                            float sum = C[ii * size + jj];
                            for (int kk = k; kk < k_end; kk++) {
                                sum += A[ii * size + kk] * B_T[jj * size + kk];
                            }
                            C[ii * size + jj] = sum;
                        }
                    }
                }
            }
        }
    } else {
        // Regular multiplication
        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < size; j++) {
                float sum = 0.0f;
                for (int k = 0; k < size; k++) {
                    sum += A[i * size + k] * B_T[j * size + k];
                }
                C[i * size + j] = sum;
            }
        }
    }

    // Terminate thread or return based on execution context
    if (USE_MULTITHREADING) {
        pthread_exit(0);
    } else {
        return NULL;
    }
}

void print_matrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            printf("%.2f ", matrix[i * size + j]);
        printf("\n");
    }
}
