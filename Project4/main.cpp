#include<cstdio>
#include<iostream>
#include"utils.h"
#include"parameters.h"

using namespace std;

// Solve the diffusion equation in 2D
int main() {

	// Timer variables
	time_type start, end;

	// Main parameters
	parameters::nx = 800;
	parameters::ny = 800;
	parameters::nz = 800;
	parameters::dx = 1. / parameters::nx;
	parameters::dy = 1. / parameters::ny;
	parameters::dz = 1. / parameters::nz;
	parameters::nt = 10;
	parameters::dt = 0.001;

	// Dirichlet Boundary Values
	parameters::uLeft = 1.0;
	parameters::uRight = 1.0;
	parameters::uBottom = 1.0;
	parameters::uTop = 1.0;
	parameters::uFront = 1.0;
	parameters::uBack = 1.0;

	// Allocating vectors
	double* u = new double[parameters::nx * parameters::ny * parameters::nz];
	double* uold = new double[parameters::nx * parameters::ny * parameters::nz];
	double* D = new double[parameters::nx * parameters::ny * parameters::nz];

	// Filling with zeros
	start = startTimer();
	fillArray(u, 0.0, parameters::nx * parameters::ny * parameters::nz);
	fillArray(uold, 0.0, parameters::nx * parameters::ny * parameters::nz);
	fillArray(D, 0.0, parameters::nx * parameters::ny * parameters::nz);
	end = endTimer(start, "fillArray");

	// Initializing the diffusion array
	start = startTimer();
	double Dconst = 0.1;
	for (int k = 0; k < parameters::nz; k++) {
		for (int j = 0; j < parameters::ny; j++) {
			for (int i = 0; i < parameters::nx; i++) {
				int indp = k * parameters::nx * parameters::ny + j * parameters::nx + i;
				D[indp] = Dconst;
			}
		}
	}
	end = endTimer(start, "Initializing diffusion Array");

	// Time loop
	double time = 0.0;
	int n;
	for (n = 0; n < parameters::nt; n++) {
		cout << "Step " << n << endl;

		// Advancing field
		start = startTimer();
		implicitDiffusionCG(u, uold, D);
		end = endTimer(start, "implicitDiffusionCG");

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