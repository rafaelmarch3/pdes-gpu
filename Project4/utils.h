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

void print3DArray(double* v, int nx, int ny, int nz, int precision = 4) {
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                cout << fixed << setprecision(precision) << v[k* nx * ny + j * nx + i] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

void writeScalarFieldToFile(const double* field, int nx, int ny, int nz, const char* filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double value = field[k * nx * ny + j * nx + i];
                file.write(reinterpret_cast<const char*>(&value), sizeof(double));
            }
        }
    }

    file.close();
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

void implicitDiffusionMatVecProduct(double* Au, const double* u, const double* D) {

    //Allocating matrix and rhs storage
    int n = parameters::nx * parameters::ny * parameters::nz;

    // Building coeff matrix and rhs
    double ay = parameters::dt / (parameters::dy * parameters::dy);
    double ax = parameters::dt / (parameters::dx * parameters::dx);
    double az = parameters::dt / (parameters::dz * parameters::dz);
    int nx = parameters::nx; int ny = parameters::ny; int nz = parameters::nz;

    for (int k = 0; k < nz; k++) {
        for (int j = 1; j < ny - 1; j++) {
            for (int i = 1; i < nx - 1; i++) {
                // Indices
                int indp = k * nx * ny + j * nx + i;
                int inde = indp + 1;
                int indw = indp - 1;
                int indn = indp + nx;
                int inds = indp - nx;
                int indf = indp - nx * ny;
                int indb = indp + nx * ny;


                // Matrix entries
                Au[indp] = (1 + 0.5 * (D[indn] + D[indp]) * ay + \
                    0.5 * (D[inds] + D[indp]) * ay + \
                    0.5 * (D[inde] + D[indp]) * ax + \
                    0.5 * (D[indw] + D[indp]) * ax + \
                    0.5 * (D[indf] + D[indp]) * az + \
                    0.5 * (D[indb] + D[indp]) * az) * u[indp] - // P
                    0.5 * (D[inde] + D[indp]) * ax * u[inde]  - // E
                    0.5 * (D[indw] + D[indp]) * ax * u[indw]  - // W
                    0.5 * (D[indn] + D[indp]) * ay * u[indn]  - // N
                    0.5 * (D[inds] + D[indp]) * ay * u[inds]  - // S
                    0.5 * (D[indf] + D[indp]) * az * u[indf]  - // F 
                    0.5 * (D[indb] + D[indp]) * az * u[indb];   // B 
            }
        }
    } // End for (Building coeff matrix and rhs)
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {

            // left
            int indp = k * nx * ny + j * nx + 0;
            Au[indp] = u[indp];

            // right
            indp = k * nx * ny + j * nx + (nx - 1);
            Au[indp] = u[indp];
        }
    }
    for (int k = 0; k < nz; k++) {
        for (int i = 0; i < nx; i++) {

            // bottom
            int indp = k * nx * ny + 0 * nx + i;
            Au[indp] = u[indp];

            //top
            indp = k * nx * ny + (ny - 1) * nx + i;
            Au[indp] = u[indp];
        }
    }
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {

            // front
            int indp = 0 * nx * ny + j * nx + i;
            Au[indp] = u[indp];

            // back
            indp = (nz - 1) * nx * ny + j * nx + i;
            Au[indp] = u[indp];
        }
    }
}

// Solves implicit diffusion using the CG method
void implicitDiffusionCG(double* u, double* uold, const double* D, int maxIterations = 150, double tolerance = 1e-4) {

    //Allocating rhs storage
    int nx = parameters::nx;
    int ny = parameters::ny;
    int nz = parameters::nz;
    int n = nx * ny * nz;
    double* rhs = new double[n];

    // RHS first has the field at the previous timestep
    copyArray(rhs, uold, n);

    // RHS also has Boundary conditions
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {

            // left
            int row = k * nx * ny + j * nx + 0;
            int colp = row;
            rhs[row] = parameters::uLeft;

            // right
            row = k * nx * ny + j * nx + (nx - 1);
            colp = row;
            rhs[row] = parameters::uRight;
        }
    }
    for (int k = 0; k < nz; k++) {
        for (int i = 0; i < nx; i++) {

            // bottom
            int row = k * nx * ny + 0 * nx + i;
            int colp = row;
            rhs[row] = parameters::uBottom; // bottom

            //top
            row = k * nx * ny + (ny - 1) * nx + i;
            colp = row;
            rhs[row] = parameters::uTop; // top
        }
    }
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {

            // front
            int row = 0 * nx * ny + j * nx + i;
            int colp = row;
            rhs[row] = parameters::uFront; // front

            // back
            row = (nz - 1) * nx * ny + j * nx + i;
            colp = row;
            rhs[row] = parameters::uBack; // back
        }
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

#endif
