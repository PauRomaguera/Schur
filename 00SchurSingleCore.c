#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <stddef.h> // 
#define ALIGN_BYTES 64  
#define MR 4
#define NR 8

//falta arreglar el kernel / padding

static void pack_blockA(const double *A, double *Atemp, int current_m, size_t ldA, size_t Kblock)
{
    for (size_t k = 0; k < Kblock; ++k) {
        for (int m = 0; m < current_m; ++m)
            Atemp[k*MR + m] = A[m*ldA + k];

        for (int m = current_m; m < MR; ++m)
            Atemp[k*MR + m] = 0.0;
    }
}

static void pack_blockB(const double *B, double *Btemp, int current_n, size_t ldB, size_t Kblock)
{
    for (size_t k = 0; k < Kblock; ++k) {
        for (int n = 0; n < current_n; ++n)
            Btemp[k*NR + n] = B[k*ldB + n];

        for (int n = current_n; n < NR; ++n)
            Btemp[k*NR + n] = 0.0;
    }
}


static void matmul_kernel_4x8(const double *__restrict__ A, size_t ldA, const double *__restrict__ B, size_t ldB, double *__restrict__ C, size_t ldC, size_t K)
{
    __m256d c0_0 = _mm256_load_pd(&C[0*ldC + 0]);
    __m256d c0_1 = _mm256_load_pd(&C[0*ldC + 4]);
    __m256d c1_0 = _mm256_load_pd(&C[1*ldC + 0]);
    __m256d c1_1 = _mm256_load_pd(&C[1*ldC + 4]);
    __m256d c2_0 = _mm256_load_pd(&C[2*ldC + 0]);
    __m256d c2_1 = _mm256_load_pd(&C[2*ldC + 4]);
    __m256d c3_0 = _mm256_load_pd(&C[3*ldC + 0]);
    __m256d c3_1 = _mm256_load_pd(&C[3*ldC + 4]);

    for (size_t k = 0; k < K; ++k) {
        __m256d b0 = _mm256_load_pd(&B[k*ldB + 0]);
        __m256d b1 = _mm256_load_pd(&B[k*ldB + 4]);

        __m256d a0 = _mm256_broadcast_sd(&A[0*ldA + k]);
        c0_0 = _mm256_fmadd_pd(a0, b0, c0_0);
        c0_1 = _mm256_fmadd_pd(a0, b1, c0_1);

        __m256d a1 = _mm256_broadcast_sd(&A[1*ldA + k]);
        c1_0 = _mm256_fmadd_pd(a1, b0, c1_0);
        c1_1 = _mm256_fmadd_pd(a1, b1, c1_1);

        __m256d a2 = _mm256_broadcast_sd(&A[2*ldA + k]);
        c2_0 = _mm256_fmadd_pd(a2, b0, c2_0);
        c2_1 = _mm256_fmadd_pd(a2, b1, c2_1);

        __m256d a3 = _mm256_broadcast_sd(&A[3*ldA + k]);
        c3_0 = _mm256_fmadd_pd(a3, b0, c3_0);
        c3_1 = _mm256_fmadd_pd(a3, b1, c3_1);
    }

    _mm256_store_pd(&C[0*ldC + 0], c0_0);
    _mm256_store_pd(&C[0*ldC + 4], c0_1);
    _mm256_store_pd(&C[1*ldC + 0], c1_0);
    _mm256_store_pd(&C[1*ldC + 4], c1_1);
    _mm256_store_pd(&C[2*ldC + 0], c2_0);
    _mm256_store_pd(&C[2*ldC + 4], c2_1);
    _mm256_store_pd(&C[3*ldC + 0], c3_0);
    _mm256_store_pd(&C[3*ldC + 4], c3_1);
}


static void matmul_edge(const double *A, const double *B, double *C, size_t ldC, int m, int n, size_t K)
{
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            double sum = C[i*ldC + j];
            for (size_t k = 0; k < K; ++k)
                sum += A[k*MR + i] * B[k*NR + j];
            C[i*ldC + j] = sum;
        }
}


void matmul_blocked(const double *A, size_t ldA, const double *B, size_t ldB, double *C, size_t ldC, size_t M, size_t N, size_t K)
{
    const size_t TM = 256, TK = 64, TN = 256;

    double *Atemp = (double*)aligned_alloc(64, MR * TK * sizeof(double));
    double *Btemp = (double*)aligned_alloc(64, NR * TK * sizeof(double));

    for (size_t i0 = 0; i0 < M; i0 += TM) {
        size_t i_end = (i0 + TM < M) ? i0 + TM : M;

        for (size_t k0 = 0; k0 < K; k0 += TK) {
            size_t k_end   = (k0 + TK < K) ? k0 + TK : K;
            size_t Kblock  = k_end - k0;

            for (size_t j0 = 0; j0 < N; j0 += TN) {
                size_t j_end = (j0 + TN < N) ? j0 + TN : N;

                for (size_t i = i0; i < i_end; i += MR) {
                    int current_m = (i + MR <= i_end) ? MR : (int)(i_end - i);

                    pack_blockA(&A[i*ldA + k0], Atemp,
                                current_m, ldA, Kblock);

                    for (size_t j = j0; j < j_end; j += NR) {
                        int current_n = (j + NR <= j_end) ? NR : (int)(j_end - j);

                        pack_blockB(&B[k0*ldB + j], Btemp,
                                    current_n, ldB, Kblock);

                        if (current_m == MR && current_n == NR) {
                            /* full 4×8 tile – use AVX micro-kernel */
                            matmul_kernel_4x8(Atemp, MR,  /* ldA = MR */
                                              Btemp, NR,  /* ldB = NR */
                                              &C[i*ldC + j], ldC,
                                              Kblock);
                        } else {
                            /* partial tile – use safe scalar kernel */
                            matmul_edge(Atemp, Btemp,
                                        &C[i*ldC + j], ldC,
                                        current_m, current_n, Kblock);
                        }
                    }
                }
            }
        }
    }
    free(Atemp);
    free(Btemp);
}



// Agafa mat A tamany N, guarda en mat A_inv tamany size
void inverse_2x2_stride(double *A, int ldA, double *A_inv, int ldInv)
{
    // printf("[DEBUG] Entering inverse_2x2_stride()\n");
    // printf("[DEBUG] ldA=%d, ldInv=%d\n", ldA, ldInv);

    // Indices basados en ldA
    double a = A[0 * ldA + 0];
    double b = A[0 * ldA + 1];
    double c = A[1 * ldA + 0];
    double d = A[1 * ldA + 1];

    double det = a * d - b * c;
    if (det == 0.0)
    {
        fprintf(stderr, "[ERROR] 2x2 block is singular.\n");
        exit(EXIT_FAILURE);
    }

    double inv_a = d / det;
    double inv_b = -b / det;
    double inv_c = -c / det;
    double inv_d = a / det;

    A_inv[0 * ldInv + 0] = inv_a;
    A_inv[0 * ldInv + 1] = inv_b;
    A_inv[1 * ldInv + 0] = inv_c;
    A_inv[1 * ldInv + 1] = inv_d;

}

void GJElimination_stride(double *A, int ldA, double *A_inv, int n)
{
    
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A_inv[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Gauss-Jordan
    for (int i = 0; i < n; i++)
    {
        double pivot = A[i * ldA + i];
        if (pivot == 0.0)
        {
            fprintf(stderr, "[ERROR] Gauss-Jordan: pivot nul a la fila %d\n", i);
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < n; j++)
        {
            A[i * ldA + j] /= pivot;
            A_inv[i * n + j] /= pivot;
        }

        for (int fila = 0; fila < n; fila++)
        {
            if (fila != i)
            {
                double pivot2 = A[fila * ldA + i];
                for (int k = 0; k < n; k++)
                {
                    A[fila * ldA + k] -= A[i * ldA + k] * pivot2;
                    A_inv[fila * n + k] -= A_inv[i * n + k] * pivot2;
                }
            }
        }
    }
}

void schur_inverse(double *A, int ldA, double *A_inv, int ldInv, int n, int base_case_size)
{
    if (n <= base_case_size) {
        GJElimination_stride(A, ldA, A_inv, n);
        return;
    }

    int half = n / 2;

    double *A11 = A;
    double *A12 = A + half;
    double *A21 = A + ldA * half;
    double *A22 = A + ldA * half + half;

    double *A11_inv = A_inv;
    double *A12_inv = A_inv + half;
    double *A21_inv = A_inv + ldInv * half;
    double *A22_inv = A_inv + ldInv * half + half;

    size_t blockSize = (size_t)half * (size_t)half * sizeof(double);

    double *R1 = NULL;
    double *R2 = NULL;
    double *R3 = NULL;
    double *R4 = NULL;
    double *R5 = NULL;

    posix_memalign((void **)&R1, ALIGN_BYTES, blockSize);
    posix_memalign((void **)&R2, ALIGN_BYTES, blockSize);
    posix_memalign((void **)&R3, ALIGN_BYTES, blockSize);
    posix_memalign((void **)&R4, ALIGN_BYTES, blockSize);
    posix_memalign((void **)&R5, ALIGN_BYTES, blockSize);
    memset(R1, 0, half * half * sizeof(double));
    memset(R2, 0, half * half * sizeof(double));
    memset(R3, 0, half * half * sizeof(double));
    memset(R4, 0, half * half * sizeof(double));
    memset(R5, 0, half * half * sizeof(double));


    //R1 = A11_inv
    schur_inverse(A11, ldA, R1, half, half, base_case_size);

    //R2 = A21 * R1 (Depends on R1)
    matmul_blocked(A21, ldA, R1, half, R2, half, half, half, half);

    //R3 = R1 * A12 (Depends on R1)
    matmul_blocked(R1, half, A12, ldA, R3, half, half, half, half);

    //R4 = A21*R3 (Depends on R3)
    matmul_blocked(A21, ldA, R3, half, R4, half, half, half, half);

    //R5 = R4 - A22 (Depends on R4)
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            R5[i * half + j] = A22[i * ldA + j] - R4[i * half + j];
        }
    }
    //R4 = R5_inv (S_inv) (Depen de R5)
    memset(R4, 0, half * half * sizeof(double));
    schur_inverse(R5, half, R4, half, half, base_case_size);

    //A22_inv = R4 
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            A22_inv[i * ldInv + j] = R4[i * half + j];
        }
    }

    matmul_blocked(R3, half, R4, half, A12_inv, ldInv, half, half, half);
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            A12_inv[i*ldInv + j] = - A12_inv[i*ldInv + j];
        }
    }
    //A21_inv = R4*R2 (Depen de R4)
    matmul_blocked(R4, half, R2, half, A21_inv, ldInv, half, half, half);
    //R5 = R3 * A21_inv
    memset(R5, 0, half * half * sizeof(double));
    matmul_blocked(R3, half, A21_inv, ldInv, R5, half, half, half, half);
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            A21_inv[i*ldInv + j] = - A21_inv[i*ldInv + j];
        }
    }
    //A11_inv = R1 + R5
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            A11_inv[i * ldInv + j] = R1[i * half + j] + R5[i * half + j];
        }
    }

    free(R1);
    free(R2);
    free(R3);
    free(R4);
    free(R5);
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Usage: %s <matrix_size> <base_case_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int base_case_size = atoi(argv[2]);

    if (N <= 0 || base_case_size <= 0)
    {
        printf("Error: Ensure matrix_size > 0, base_case_size > 0, and matrix_size is divisible by base_case_size.\n");
        return 1;
    }
    double *A = (double *)malloc(N * N * sizeof(double));
    double *A_inv = (double *)malloc(N * N * sizeof(double));
    memset(A_inv, 0, N * N * sizeof(double));
    srand(0u);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i * N + j] = ((double)rand() / RAND_MAX);
    printf("Print A:\n");
    schur_inverse(A, N, A_inv, N, N, base_case_size);
    double sum = 0.0f;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            sum += A_inv[i * N + j];
    free(A);
    free(A_inv);
    printf("Checksum A:%.8f \n", sum);

    return 0;
}
