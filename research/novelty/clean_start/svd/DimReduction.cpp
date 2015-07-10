/* Author: Marlos C. Machado */

#include <iostream>

#include "DimReduction.hpp"

using namespace std;
using namespace Eigen;

void centerMatrix(MatrixXi dataset, MatrixXf &centeredDataset){}

void fillCenteringVector(vector<float> &centeringVector){}

void fillWithTopEigenVectors(int k, vector<vector<float> > &eigenVectors){}

JacobiSVD<MatrixXf> reduceDimensionalityOfEvents(MatrixXi dataset, 
	vector<float> &centeringVector, vector<vector<float> > &eigenVectors, int k){

	MatrixXf centeredDataset;	
	centerMatrix(dataset, centeredDataset);
	//Proper SVD:
	JacobiSVD<MatrixXf> svd(centeredDataset, ComputeThinU | ComputeThinV);
	fillWithTopEigenVectors(k, eigenVectors);
	fillCenteringVector(centeringVector);
	
	return svd;
}