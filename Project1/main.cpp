#include <cstdio>
#include <math.h>
#include <fstream>
#include <string>
#include <iostream>

using namespace std;

#define SIZE_X 10 // Number of grid points in the x-direction
#define SIZE_Y 10 // Number of grid points in the y-direction
#define SIZE_Z 10 // Number of grid points in the z-direction
#define MAX_ITER 1000 // Maximum number of iterations
#define TOLERANCE 1e-6 // Tolerance for convergence

double A[SIZE_X][SIZE_Y][SIZE_Z]; // Coefficient matrix
double b[SIZE_X][SIZE_Y][SIZE_Z]; // Right-hand side vector
double x[SIZE_X][SIZE_Y][SIZE_Z]; // Solution vector
double r[SIZE_X][SIZE_Y][SIZE_Z]; // Residual vector
double p[SIZE_X][SIZE_Y][SIZE_Z]; // Search direction vector
double Ap[SIZE_X][SIZE_Y][SIZE_Z]; // Matrix-vector product

// Function to compute the matrix-vector product Ap
void matVecProduct(double Ap[SIZE_X][SIZE_Y][SIZE_Z], double x[SIZE_X][SIZE_Y][SIZE_Z]) {
    int i, j, k;
    double h = 1.0; // Assuming uniform grid spacing

    // Compute Ap using the discretized formula

    for (i = 0; i < SIZE_X; i++) {
        for (j = 0; j < SIZE_Y; j++) {
            for (k = 0; k < SIZE_Z; k++) {
                Ap[i][j][k] = (1 + 6 * h * h) * x[i][j][k];
                if (i > 0) Ap[i][j][k] -= x[i - 1][j][k];
                if (i < SIZE_X - 1) Ap[i][j][k] -= x[i + 1][j][k];
                if (j > 0) Ap[i][j][k] -= x[i][j - 1][k];
                if (j < SIZE_Y - 1) Ap[i][j][k] -= x[i][j + 1][k];
                if (k > 0) Ap[i][j][k] -= x[i][j][k - 1];
                if (k < SIZE_Z - 1) Ap[i][j][k] -= x[i][j][k + 1];
            }
        }
    }
}

// Function to compute the dot product of two vectors
double dotProduct(double a[SIZE_X][SIZE_Y][SIZE_Z], double b[SIZE_X][SIZE_Y][SIZE_Z]) {
    int i, j, k;
    double result = 0.0;

    // Compute dot product
    for (i = 0; i < SIZE_X; i++) {
        for (j = 0; j < SIZE_Y; j++) {
            for (k = 0; k < SIZE_Z; k++) {
                result += a[i][j][k] * b[i][j][k];
            }
        }
    }

    return result;
}

// Function to perform the Conjugate Gradient method
void conjugateGradient() {
    int iter;
    double alpha, beta, residual;

    // Initialize solution vector x with zeros
    // Initialize other vectors as needed
    int i, j, k;
    for (i = 0; i < SIZE_X; i++) {
        for (j = 0; j < SIZE_Y; j++) {
            for (k = 0; k < SIZE_Z; k++) {
                x[i][j][k] = 0.0;
                r[i][j][k] = b[i][j][k];
                p[i][j][k] = r[i][j][k];
            }
        }
    }

    for (iter = 0; iter < MAX_ITER; iter++) {
        // Compute matrix-vector product Ap
        matVecProduct(Ap, p);

        // Compute alpha
        alpha = dotProduct(r, r) / dotProduct(p, Ap);

        // Update solution vector x
        for (i = 0; i < SIZE_X; i++) {
            for (j = 0; j < SIZE_Y; j++) {
                for (k = 0; k < SIZE_Z; k++) {
                    x[i][j][k] += alpha * p[i][j][k];
                }
            }
        }

        // Update residual vector r
        for (i = 0; i < SIZE_X; i++) {
            for (j = 0; j < SIZE_Y; j++) {
                for (k = 0; k < SIZE_Z; k++) {
                    r[i][j][k] -= alpha * Ap[i][j][k];
                }
            }
        }

        // Compute residual norm
        residual = sqrt(dotProduct(r, r));

        // Check for convergence
        if (residual < TOLERANCE) {
            printf("Convergence achieved after %d iterations.\n", iter + 1);
            break;
        }

        // Compute beta
        beta = dotProduct(r, r) / dotProduct(p, p);

        // Update search direction p
        for (i = 0; i < SIZE_X; i++) {
            for (j = 0; j < SIZE_Y; j++) {
                for (k = 0; k < SIZE_Z; k++) {
                    p[i][j][k] = r[i][j][k] + beta * p[i][j][k];
                }
            }
        }
    }

    if (iter == MAX_ITER) {
        printf("Convergence not achieved within maximum iterations.\n");
    }
}


void write_to_file(int nx, int ny, int nz, double*** data, string path) {

    string fullpath_bin = path + "output.bin";
    string fullpath_bov = path + "output.bov";

    FILE* output = fopen(fullpath_bin.c_str(), "w");
    fwrite(data, sizeof(double), nx * ny * nz, output);
    fclose(output);

    std::ofstream fid(fullpath_bov.c_str());
    fid << "TIME: 0.0" << std::endl;
    fid << "DATA_FILE: output.bin" << std::endl;
    fid << "DATA_SIZE: " << nx << " " << ny << " 1" << std::endl;;
    fid << "DATA_FORMAT: DOUBLE" << std::endl;
    fid << "VARIABLE: phi" << std::endl;
    fid << "DATA_ENDIAN: LITTLE" << std::endl;
    fid << "CENTERING: nodal" << std::endl;
    fid << "BRICK_SIZE: 1.0 1.0 1.0" << std::endl;
}


int main() {
    /*
    // Initialize coefficient matrix A and right-hand side vector b
    int i, j, k;
    for (i = 0; i < SIZE_X; i++) {
        for (j = 0; j < SIZE_Y; j++) {
            for (k = 0; k < SIZE_Z; k++) {
                A[i][j][k] = 1 + 6 * pow(1.0, 2); // Assuming uniform grid spacing of 1
                b[i][j][k] = 0.0; // Zero initial condition
            }
        }
    }

    // Set boundary conditions
    for (j = 0; j < SIZE_Y; j++) {
        for (k = 0; k < SIZE_Z; k++) {
            b[0][j][k] = 1.0; // Dirichlet boundary condition of 1 at x = 0
            b[SIZE_X - 1][j][k] = 1.0; // Dirichlet boundary condition of 1 at x = SIZE_X - 1
        }
    }

    for (i = 0; i < SIZE_X; i++) {
        for (j = 0; j < SIZE_Y; j++) {
            b[i][j][0] = 1.0; // Dirichlet boundary condition of 1
            b[i][j][SIZE_Z - 1] = 1.0; // Dirichlet boundary condition of 1
        }
    }
    // Call conjugateGradient function
    conjugateGradient();

    // Output the solution vector x
    printf("Solution vector x:\n");
    for (i = 0; i < SIZE_X; i++) {
        for (j = 0; j < SIZE_Y; j++) {
            for (k = 0; k < SIZE_Z; k++) {
                printf("%f ", x[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Writing output file
    string path = "D:/personal/studies/pdes-modern-cpp-and-cuda/pdes-gpu/out/";

    write_to_file(SIZE_X, SIZE_Y, SIZE_Z, x, path);

    */
    return 0;
}