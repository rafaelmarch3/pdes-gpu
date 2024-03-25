#include<cstdio>
#include<iostream>
#include"utils.h"
#include"parameters.h"

using namespace std;

// Solve the diffusion equation in 2D
int main() {

	// Main parameters
	parameters::nx = 100;
	parameters::ny = 100;
	parameters::dx = 1;
	parameters::dy = 1;
	parameters::nt = 10;
	parameters::dt = 50;

	// Dirichlet Boundary Values
	parameters::uLeft = 1.;
	parameters::uRight = 1.;
	parameters::uBottom = 1.;
	parameters::uTop = 1.;

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

	// Printing Diffusion
	//print2DArray(D, parameters::nx, parameters::ny, 8);
	//cout << "===" << endl;

	/*
	// Testing conjugate gradient
	int ntest = 5;
	double* Atest = new double[ntest * ntest];
	double* btest = new double[ntest];
	double* Axtest = new double[ntest];
	double* xtest = new double[ntest];
	fillArray(Atest, 0.0, ntest * ntest);
	fillArray(Axtest, 0.0, ntest);
	fillArray(btest, 1.0, ntest);
	fillArray(xtest, 0.0, ntest);

	for (int j = 0; j < ntest; j++) {
		for (int i = 0; i < ntest; i++) {
			Atest[j * ntest + i] = i + j + 1;
		}
	}
	print2DArray(Atest, ntest, ntest);
	conjugateGradient(Atest, btest, xtest, ntest);

	print2DArray(xtest, 1, ntest);
	cout << "--" << endl;
	matVecProduct(Atest, xtest, Axtest, ntest);
	print2DArray(Axtest, 1, ntest);
	*/

	// Time loop
	string outFolder = "D:/personal/studies/pdes-modern-cpp-and-cuda/pdes-gpu/out/";
	double time = 0.0;
	int n;
	for (n = 0; n < parameters::nt; n++) {
		cout << "Step " << n << endl;
		
		// Advancing field
		//explicitDiffusion(u, uold, D);

		// Advancing field
		implicitDiffusion(u, uold, D);

		// Printing Array
		//print2DArray(u, parameters::nx, parameters::ny);
		cout << "---" << endl;

		// Updating uold
		copyArray(uold, u, parameters::nx * parameters::ny);

		// Writing binary file
		/*
		string prefix = "field_" + to_string(n + 1);
		writeFieldBinary(u,
			parameters::nx*parameters::ny,
			outFolder,
			prefix);
			*/
		// Writing bov file
		
	}

	//string prefix = "field_" + to_string(n + 1);
	string prefix = "field_final";
	string fullpath_bin = outFolder + prefix + ".bin";
	string fullpath_bov = outFolder + prefix + ".bov";
	writeScalarFieldToFile(u,
		parameters::nx, parameters::ny,
		fullpath_bin.c_str());

	/*
	writeBOVFile(parameters::nx, parameters::ny,
		parameters::dx, parameters::dy,
		(n + 1) * parameters::dt,
		fullpath_bin.c_str(),
		fullpath_bov.c_str());
		*/


	
	return 0;
}