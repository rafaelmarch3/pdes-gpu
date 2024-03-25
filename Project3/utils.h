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

void print2DArray(double* v, int nx, int ny, int precision = 4) {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            cout << fixed << setprecision(precision) << v[j * nx + i] << " ";
            //cout << v[j * nx + i] << " ";
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

void writeBOVFile(int nx, int ny, int dx, int dy, double time, const char* dataFileName, const char* bovFileName) {
    std::ofstream file(bovFileName);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << bovFileName << " for writing." << std::endl;
        return;
    }

    file << "TIME: " << time << std::endl;
    file << "DATA_FILE: " << dataFileName << std::endl;
    file << "DATA_SIZE: " << nx << " " << ny << " 1" << std::endl;
    file << "DATA_FORMAT: DOUBLE" << std::endl;
    file << "VARIABLE: SCALAR" << std::endl;
    file << "DATA_ENDIAN: LITTLE_ENDIAN" << std::endl;
    file << "CENTERING: ZONAL" << std::endl;
    file << "BRICK_ORIGIN: 0. 0. 0." << std::endl;
    file << "BRICK_SIZE: " << dx << " " << dy << " 1" << std::endl;


    /*
    * TIME: 10.
DATA_FILE: density.bof
DATA_SIZE: 10 10 10
DATA_FORMAT: FLOAT
VARIABLE: density
DATA_ENDIAN: LITTLE
CENTERING: ZONAL
BRICK_ORIGIN: 0. 0. 0.
BRICK_SIZE: 10. 10. 10.
    */

    file.close();
}

/*
void writeFieldBOV(double* field,
                double time,
                int nx, int ny, int nz,
                double dx, double dy, double dz,
                string target_folder, 
                string prefix) {

    string fullpath_bin = target_folder + prefix + ".bin";
    string fullpath_bov = target_folder + prefix + ".bov";

    auto myfile = fstream(fullpath_bin,
        std::ios::out | std::ios::binary);
    myfile.write((char*)&field[0], nx * ny * nz);
    myfile.close();
   
    //FILE* output = fopen(fullpath_bin.c_str(), "w");
    //fwrite(field, sizeof(double), nx * ny * nz, output);
    //fclose(output);
    

    std::ofstream fid(fullpath_bov.c_str());
    fid << "TIME: " << to_string(time) << std::endl;
    fid << "DATA_FILE: " << fullpath_bin << std::endl;
    fid << "DATA_SIZE: " << nx << " " << ny << " " << nz << std::endl;
    fid << "DATA_FORMAT: DOUBLE" << std::endl;
    fid << "VARIABLE: phi" << std::endl;
    fid << "DATA_ENDIAN: LITTLE" << std::endl;
    fid << "CENTERING: nodal" << std::endl;
    fid << "BRICK_SIZE: " << dx << " " << dy << " " << dz << std::endl;
}

void writeFieldBinary(double* field,
                    int size,
                    string target_folder,
                    string prefix) {

    string fullpath_bin = target_folder + prefix + ".bin";
    auto myfile = fstream(fullpath_bin,
        std::ios::out | std::ios::binary);
    myfile.write((char*)&field[0], size);
    myfile.close();

}
*/

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

// Conjugate gradient method to solve Ax = b
void conjugateGradient(const double* A, const double* b, double* x, int N, int maxIterations=150, double tolerance=1e-8) {
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

    cout << "CG Residual = " << dotProduct(r, r, N) << endl;
    // Main loop of conjugate gradient method
    for (int iter = 0; iter < maxIterations; ++iter) {
        matVecProduct(A, p, Ap, N); // Compute Ap
        double alpha = dotProduct(r, r, N) / dotProduct(p, Ap, N); // Compute step size

        // Update solution vector x and residual vector r
        for (int i = 0; i < N; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        cout << "CG Residual = " << dotProduct(r, r, N) << endl;
        // Check for convergence
        if (sqrt(dotProduct(r, r, N)) < tolerance) {
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

    // Free allocated memory
    delete[] r;
    delete[] p;
    delete[] Ap;
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
