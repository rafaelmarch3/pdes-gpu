#ifndef __UTILS_H
#define __UTILS_H

#pragma warning(disable:4996)
#pragma warning(disable:6386)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <string>
#include <iostream>
#include<iomanip>
#include"parameters.h"

using namespace std;


void fillArray(double* v, double val, int size) {
    for (int j = 0; j < size; j++) {
        v[j] = val;
    }
}

void copyArray(double* dest, const double* src, int size) {
    for (int j = 0; j < size; j++) {
        dest[j] = src[j];
    }
}

void print1DArray(double* v, int n, int precision = 4) {
    for (int j = 0; j < n; j++) {
        cout << fixed << setprecision(precision) << v[j] << " ";
    }
}

void print2DArray(double* v, int nx, int ny, int precision = 4) {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            cout << fixed << setprecision(precision) << v[j * nx + i] << " ";
        }
        cout << endl;
    }
}

void writeScalarFieldToFile(const double* field, int nx, int ny, const char* filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            double value = field[j * nx + i];
            file.write(reinterpret_cast<const char*>(&value), sizeof(double));
        }
    }

    file.close();
}

// Function to compute the matrix-vector product Ax
void matVecProduct(const double* A, const double* x, double* Ax, int N) {
    // Perform the matrix-vector product
    for (int i = 0; i < N; ++i) {
        Ax[i] = 0.0;
        for (int j = 0; j < N; ++j) {
            Ax[i] += A[i * N + j] * x[j];
        }
    }
}

// Function to compute the dot product of two vectors
double dotProduct(const double* a, const double* b, int N) {
    double result = 0.0;
    for (int i = 0; i < N; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// computes y := alpha*x + y
// x and y are vectors on length N
// alpha is a scalar
void axpy(double* y, const double alpha, const double* x, const int N)
{
    for (int i = 0; i < N; i++)
        y[i] += alpha * x[i];
}

// computes y = x + alpha*(l-r)
// y, x, l and r are vectors of length N
// alpha is a scalar
void addScaledDiff(double* y, const double* x, const double alpha,
    const double* l, const double* r, const int N)
{
    for (int i = 0; i < N; i++)
        y[i] = x[i] + alpha * (l[i] - r[i]);
}

// computes y = alpha*(l-r)
// y, l and r are vectors of length N
// alpha is a scalar
void scaleDiff(double* y, const double alpha,
    const double* l, const double* r, const int N)
{
    for (int i = 0; i < N; i++)
        y[i] = alpha * (l[i] - r[i]);
}

// computes y := alpha*x
// alpha is scalar
// y and x are vectors on length n
void scale(double* y, const double alpha, double* x, const int N)
{
    for (int i = 0; i < N; i++)
        y[i] = alpha * x[i];
}

// computes linear combination of two vectors y := alpha*x + beta*z
// alpha and beta are scalar
// y, x and z are vectors on length n
void linearCombination(double* y, const double alpha, double* x, const double beta,
    const double* z, const int N)
{
    for (int i = 0; i < N; i++)
        y[i] = alpha * x[i] + beta * z[i];
}

// Conjugate gradient method to solve Ax = b
void conjugateGradient(const double* A, const double* b, double* x, int N, int maxIterations=150, double tolerance=1e-4) {
    double* r = new double[N]; // Residual vector
    double* p = new double[N]; // Search direction vector
    double* Ap = new double[N]; // Matrix-vector product

    // Initialize solution vector x with zeros
    for (int i = 0; i < N; ++i) {
        x[i] = 0.0;
    }

    // Compute initial residual r = b - Ax
    matVecProduct(A, x, Ap, N); // Compute Ax
    for (int i = 0; i < N; ++i) {
        r[i] = b[i] - Ap[i];
        p[i] = r[i]; // Set initial search direction as residual
    }

    cout << "Starting CG Residual = " << dotProduct(r, r, N) << endl;
    // Main loop of conjugate gradient method
    int iter = 0;
    for (iter = 0; iter < maxIterations; ++iter) {
        matVecProduct(A, p, Ap, N); // Compute Ap
        double alpha = dotProduct(r, r, N) / dotProduct(p, Ap, N); // Compute step size

        // Update solution vector x and residual vector r
        for (int i = 0; i < N; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        //cout << "CG Residual = " << dotProduct(r, r, N) << endl;
        // Check for convergence
        if (sqrt(dotProduct(r, r, N)) < tolerance) {
            cout << "Final CG Residual = " << dotProduct(r, r, N) << endl;
            cout << "Convergence achieved after " << iter + 1 << " iterations." << endl;
            break;
        }

        // Compute beta for next iteration
        double beta = dotProduct(r, r, N) / dotProduct(p, Ap, N);

        // Update search direction vector p
        for (int i = 0; i < N; ++i) {
            p[i] = r[i] + beta * p[i];
        }
    }

    if (iter == maxIterations) {
        cout << "Final CG Residual = " << dotProduct(r, r, N) << endl;
        cout << "No Convergence achieved after " << iter << " iterations." << endl;
    }

    // Free allocated memory
    delete[] r;
    delete[] p;
    delete[] Ap;
}

void implicitDiffusionMatVecProduct(double* Au, const double* u, const double* D) {

    //Allocating matrix and rhs storage
    int n = parameters::nx * parameters::ny;

    // Building coeff matrix and rhs
    double ay = parameters::dt / (parameters::dy * parameters::dy);
    double ax = parameters::dt / (parameters::dx * parameters::dx);
    int nx = parameters::nx; int ny = parameters::ny;
    for (int j = 1; j < parameters::ny - 1; j++) {
        for (int i = 1; i < parameters::nx - 1; i++) {
            // Indices
            int indp = j * nx + i;
            int inde = indp + 1;
            int indw = indp - 1;
            int indn = indp + nx;
            int inds = indp - nx;

            // Matrix entries
            Au[indp] = (1 + 0.5 * (D[indn] + D[indp]) * ay + \
                            0.5 * (D[inds] + D[indp]) * ay + \
                            0.5 * (D[inde] + D[indp]) * ax + \
                            0.5 * (D[indw] + D[indp]) * ax) * u[indp] - // P
                            0.5 * (D[inde] + D[indp]) * ax  * u[inde] - // E
                            0.5 * (D[indw] + D[indp]) * ax  * u[indw] - // W
                            0.5 * (D[indn] + D[indp]) * ay  * u[indn] - // N
                            0.5 * (D[inds] + D[indp]) * ay  * u[inds];  // S
        }
    } // End for (Building coeff matrix and rhs)

    for (int j = 0; j < ny; j++) {

        // left
        int indp = j * nx + 0;
        Au[indp] = u[indp];

        // right
        indp = j * nx + (nx - 1);
        Au[indp] = u[indp];
    }

    for (int i = 0; i < nx; i++) {

        // bottom
        int indp = 0 * nx + i;
        Au[indp] = u[indp];

        //top
        indp = (ny - 1) * nx + i;
        Au[indp] = u[indp];
    }

}

// Solves implicit diffusion using the CG method
void implicitDiffusionCG(double* u, double* uold, const double* D, int maxIterations = 150, double tolerance = 1e-4) {

    //Allocating rhs storage
    int nx = parameters::nx;
    int ny = parameters::ny;
    int n = nx * ny;
    double* rhs = new double[n];

    // RHS first has the field at the previous timestep
    copyArray(rhs, uold, n);

    // RHS also has Boundary conditions
    for (int j = 0; j < ny; j++) {

        // left
        int row = j * nx + 0;
        int colp = row;
        rhs[row] = parameters::uLeft;

        // right
        row = j * nx + (nx - 1);
        colp = row;
        rhs[row] = parameters::uRight;
    }
    for (int i = 0; i < nx; i++) {

        // bottom
        int row = 0 * nx + i;
        int colp = row;
        rhs[row] = parameters::uBottom; // bottom

        //top
        row = (ny - 1) * nx + i;
        colp = row;
        rhs[row] = parameters::uTop; // top
    }

    double* r = new double[n]; // Residual vector
    double* p = new double[n]; // Search direction vector
    double* Ap = new double[n]; // Matrix-vector product

    // Initialize solution vector u with zeros 
    fillArray(u, 0.0, n);

    // Compute initial residual r = b - Ax
    implicitDiffusionMatVecProduct(Ap, u, D); // Compute Au

    // r := rhs - Ap
    linearCombination(r, 1.0, rhs, -1.0, Ap, n);

    // p := r
    copyArray(p, r, n);
    
    cout << "Starting CG Residual = " << dotProduct(r, r, n) << endl;
    // Main loop of conjugate gradient method
    int iter = 0;
    for (iter = 0; iter < maxIterations; ++iter) {
        implicitDiffusionMatVecProduct(Ap, p, D);
        double alpha = dotProduct(r, r, n) / dotProduct(p, Ap, n); // Compute step size

        // u := u + alpha*p
        axpy(u, alpha, p, n);

        // r := r - alpha*Ap
        axpy(r, -alpha, Ap, n);

        //cout << "CG Residual = " << dotProduct(r, r, N) << endl;
        // Check for convergence
        if (sqrt(dotProduct(r, r, n)) < tolerance) {
            cout << "Final CG Residual = " << dotProduct(r, r, n) << endl;
            cout << "Convergence achieved after " << iter + 1 << " iterations." << endl;
            break;
        }

        // Compute beta for next iteration
        double beta = dotProduct(r, r, n) / dotProduct(p, Ap, n);

        // p := r + beta*p
        linearCombination(p, 1.0, r, beta, p, n);
    }

    if (iter == maxIterations) {
        cout << "Final CG Residual = " << dotProduct(r, r, n) << endl;
        cout << "No Convergence achieved after " << iter << " iterations." << endl;
    }

    // Free allocated memory
    delete[] r;
    delete[] p;
    delete[] Ap;
    /* */

}

void explicitDiffusion(double* u, double* uold, const double* D) {

    // Helper variables
    double ay = parameters::dt / (parameters::dy * parameters::dy);
    double ax = parameters::dt / (parameters::dx * parameters::dx);
    double Dn = 0.0;
    double Ds = 0.0;
    double De = 0.0;
    double Dw = 0.0;

    // Interior Points
    for (int j = 1; j < parameters::ny-1; j++) {
        for (int i = 1; i < parameters::nx-1; i++) {

            // Indices
            int indP = j * parameters::nx + i;
            int indE = indP + 1;
            int indW = indP - 1;
            int indN = indP + parameters::nx;
            int indS = indP - parameters::nx;

            // Diffusion values
            Dn = 0.5 * (D[indN] + D[indP]);
            Ds = 0.5 * (D[indS] + D[indP]);
            De = 0.5 * (D[indE] + D[indP]);
            Dw = 0.5 * (D[indW] + D[indP]);

            // Compute u
            u[indP] = uold[indP] +  Dn * ay * (uold[indN] - uold[indP]) -
                Ds * ay * (uold[indP] - uold[indS]) +
                De * ax * (uold[indE] - uold[indP]) -
                Dw * ax * (uold[indP] - uold[indW]);
        }
    }

    // Left boundary
    for (int j = 1; j < parameters::ny - 1; j++) {
        int i = 0;
        // Indices
        int indP = j * parameters::nx + i;
        int indE = indP + 1;
        int indN = indP + parameters::nx;
        int indS = indP - parameters::nx;

        // Diffusion values
        Dn = 0.5 * (D[indN] + D[indP]);
        Ds = 0.5 * (D[indS] + D[indP]);
        De = 0.5 * (D[indE] + D[indP]);
        Dw = D[indP]; // We mirror the diffusion coefficient

        // Compute u
        u[indP] = uold[indP] + Dn * ay * (uold[indN] - uold[indP]) -
            Ds * ay * (uold[indP] - uold[indS]) +
            De * ax * (uold[indE] - uold[indP]) -
            Dw * ax * (uold[indP] - parameters::uLeft);
    }
    // Right boundary
    for (int j = 1; j < parameters::ny - 1; j++) {
        int i = parameters::nx - 1;
        // Indices
        int indP = j * parameters::nx + i;
        int indW = indP - 1;
        int indN = indP + parameters::nx;
        int indS = indP - parameters::nx;

        // Diffusion values
        Dn = 0.5 * (D[indN] + D[indP]);
        Ds = 0.5 * (D[indS] + D[indP]);
        De = D[indP]; // We mirror the diffusion coefficient
        Dw = 0.5 * (D[indW] + D[indP]);

        // Compute u
        u[indP] = uold[indP] + Dn * ay * (uold[indN] - uold[indP]) -
            Ds * ay * (uold[indP] - uold[indS]) +
            De * ax * (parameters::uRight - uold[indP]) -
            Dw * ax * (uold[indP] - uold[indW]);
    }
    // Top boundary
    for (int i = 1; i < parameters::nx - 1; i++) {
        int j = parameters::ny - 1;
        // Indices
        int indP = j * parameters::nx + i;
        int indE = indP + 1;
        int indW = indP - 1;
        int indS = indP - parameters::nx;

        // Diffusion values
        Dn = D[indP]; // We mirror the diffusion coefficient
        Ds = 0.5 * (D[indS] + D[indP]);
        De = 0.5 * (D[indE] + D[indP]);
        Dw = 0.5 * (D[indW] + D[indP]);

        // Compute u
        u[indP] = uold[indP] + Dn * ay * (parameters::uTop - uold[indP]) -
            Ds * ay * (uold[indP] - uold[indS]) +
            De * ax * (uold[indE] - uold[indP]) -
            Dw * ax * (uold[indP] - uold[indW]);
    }
    // Bottom boundary
    for (int i = 1; i < parameters::nx - 1; i++) {
        int j = 0;
        // Indices
        int indP = j * parameters::nx + i;
        int indE = indP + 1;
        int indW = indP - 1;
        int indN = indP + parameters::nx;

        // Diffusion values
        Dn = 0.5 * (D[indN] + D[indP]);
        Ds = D[indP]; // We mirror the diffusion coefficient
        De = 0.5 * (D[indE] + D[indP]);
        Dw = 0.5 * (D[indW] + D[indP]);

        // Compute u
        u[indP] = uold[indP] + Dn * ay * (uold[indN] - uold[indP]) -
            Ds * ay * (uold[indP] - parameters::uBottom) +
            De * ax * (uold[indE] - uold[indP]) -
            Dw * ax * (uold[indP] - uold[indW]);
    }
}

void implicitDiffusion(double* u, double* uold, const double* D) {
    
    //Allocating matrix and rhs storage
    int n = parameters::nx * parameters::ny;
    double* Adiffusion = new double[n * n];
    double* bdiffusion = new double[n];
    fillArray(Adiffusion, 0.0, n * n);
    fillArray(bdiffusion, 0.0, n );

    // Building coeff matrix and rhs
    double ay = parameters::dt / (parameters::dy * parameters::dy);
    double ax = parameters::dt / (parameters::dx * parameters::dx);
    int nx = parameters::nx; int ny = parameters::ny;
    for (int j = 1; j < parameters::ny-1; j++) {
        for (int i = 1; i < parameters::nx-1; i++) {

            // Indices
            int row = j * nx + i;
            int colp = row;
            int cole = row + 1;
            int colw = row - 1;
            int coln = row + nx;
            int cols = row - nx;

            // Matrix entries
            Adiffusion[row * n + colp] = 1 + 0.5 * (D[j * nx + i] + D[j * nx + i])  * ay + \
                                         0.5 * (D[j * nx + i - nx] + D[j * nx + i]) * ay + \
                                         0.5 * (D[j * nx + i + 1] + D[j * nx + i])  * ax + \
                                         0.5 * (D[j * nx + i - 1] + D[j * nx + i])  * ax; // P
            Adiffusion[row * n + cole]  = -0.5 * (D[j * nx + i + 1] + D[j * nx + i]) * ax; // E
            Adiffusion[row * n + colw]  = -0.5 * (D[j * nx + i - 1] + D[j * nx + i]) * ax; // W
            Adiffusion[row * n + coln] = -0.5 * (D[j * nx + i + nx] + D[j * nx + i]) * ay; // N
            Adiffusion[row * n + cols] = -0.5 * (D[j * nx + i - nx] + D[j * nx + i]) * ay; // S

            // RHS
            bdiffusion[row] = uold[j * nx + i];

        }
    } // End for (Building coeff matrix and rhs)
    
    for (int j = 0; j < ny; j++) {
        // left
        int row = j * nx + 0;
        int colp = row;
        Adiffusion[row * n + colp] = 1.0;
        bdiffusion[row] = parameters::uLeft; 
        // right
        row = j * nx + (nx - 1);
        colp = row;
        Adiffusion[row * n + colp] = 1.0; 
        bdiffusion[row] = parameters::uRight; 
    }
    for (int i = 0; i < nx; i++) {
        // bottom
        int row = 0 * nx + i;
        int colp = row;
        Adiffusion[row * n + colp] = 1.0; // bottom
        bdiffusion[row] = parameters::uBottom; // bottom
        //top
        row = (ny - 1) * nx + i;
        colp = row;
        Adiffusion[row * n + colp] = 1.0; // top
        bdiffusion[row] = parameters::uTop; // top
    }
    

    conjugateGradient(Adiffusion, bdiffusion, u, n);
        
}

#endif
