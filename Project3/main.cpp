#include<cstdio>
#include<iostream>
#include"utils.h"
#include"parameters.h"

using namespace std;

// Solve the diffusion equation in 2D
int main() {

	// Main parameters
	parameters::nx = 1000;
	parameters::ny = 1000;
	parameters::dx = 1. / parameters::nx;
	parameters::dy = 1. / parameters::ny;
	parameters::nt = 10;
	parameters::dt = 0.001;

	// Dirichlet Boundary Values
	parameters::uLeft = 2.5;
	parameters::uRight = 2.5;
	parameters::uBottom = 2.5;
	parameters::uTop = 2.5;

	// Allocating vectors
	double* u = new double[parameters::nx * parameters::ny];
	double* uold = new double[parameters::nx * parameters::ny];
	double* D = new double[parameters::nx * parameters::ny];

	// Filling with zeros
	fillArray(u, 0.0, parameters::nx * parameters::ny);
	fillArray(uold, 0.0, parameters::nx * parameters::ny);
	fillArray(D, 0.0, parameters::nx * parameters::ny);

	// Initializing the diffusion array
	double Dconst = 0.1;
	for (int j = 0; j < parameters::ny; j++) {
		for (int i = 0; i < parameters::nx; i++) {
			D[j * parameters::nx + i] = Dconst;
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
		copyArray(uold, u, parameters::nx * parameters::ny);

	}

	string outFolder = "D:/personal/studies/pdes-modern-cpp-and-cuda/pdes-gpu/out/";
	string prefix = "field_final";
	string fullpath_bin = outFolder + prefix + ".bin";
	string fullpath_bov = outFolder + prefix + ".bov";
	writeScalarFieldToFile(u,
		parameters::nx, parameters::ny,
		fullpath_bin.c_str());

	
	return 0;
}