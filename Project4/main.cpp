#include<cstdio>
#include<iostream>
#include"utils.h"
#include"parameters.h"

using namespace std;

// Solve the diffusion equation in 2D
int main() {

	// Main parameters
	parameters::nx = 10;
	parameters::ny = 10;
	parameters::nz = 10;
	parameters::dx = 1. / parameters::nx;
	parameters::dy = 1. / parameters::ny;
	parameters::dz = 1. / parameters::nz;
	parameters::nt = 10;
	parameters::dt = 0.1;

	// Dirichlet Boundary Values
	parameters::uLeft = 2.5;
	parameters::uRight = 2.5;
	parameters::uBottom = 2.5;
	parameters::uTop = 2.5;
	parameters::uFront = 2.5;
	parameters::uBack = 2.5;

	// Allocating vectors
	double* u = new double[parameters::nx * parameters::ny * parameters::nz];
	double* uold = new double[parameters::nx * parameters::ny * parameters::nz];
	double* D = new double[parameters::nx * parameters::ny * parameters::nz];

	// Filling with zeros
	fillArray(u, 0.0, parameters::nx * parameters::ny * parameters::nz);
	fillArray(uold, 0.0, parameters::nx * parameters::ny * parameters::nz);
	fillArray(D, 0.0, parameters::nx * parameters::ny * parameters::nz);

	// Initializing the diffusion array
	double Dconst = 0.1;
	for (int k = 0; k < parameters::nz; k++) {
		for (int j = 0; j < parameters::ny; j++) {
			for (int i = 0; i < parameters::nx; i++) {
				int indp = k * parameters::nx * parameters::ny + j * parameters::nx + i;
				D[indp] = Dconst;
			}
		}
	}

	// Time loop
	double time = 0.0;
	int n;
	for (n = 0; n < parameters::nt; n++) {
		cout << "Step " << n << endl;

		// Advancing field
		implicitDiffusionCG(u, uold, D);

		// Updating uold
		copyArray(uold, u, parameters::nx * parameters::ny * parameters::nz);

	}

	string outFolder = "D:/personal/studies/pdes-modern-cpp-and-cuda/pdes-gpu/out/";
	string prefix = "field_final";
	string fullpath_bin = outFolder + prefix + ".bin";
	string fullpath_bov = outFolder + prefix + ".bov";
	writeScalarFieldToFile(u,
		parameters::nx, parameters::ny, parameters::nz,
		fullpath_bin.c_str());


	return 0;
}