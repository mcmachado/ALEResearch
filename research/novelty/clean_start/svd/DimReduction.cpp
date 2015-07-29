/* Author: Marlos C. Machado */

#include <fstream>
#include <iostream>

#include "DimReduction.hpp"

using namespace std;
using namespace Eigen;

void obtainStatistics(MatrixXi dataset, vector<float> &datasetMeans, 
	vector<float> &datasetStds){

	float mean, error, sumSquaredErrors;
	float numRows = dataset.rows();

	for(int i = 0; i < dataset.cols(); i++){
		mean = dataset.col(i).sum()/numRows;
		datasetMeans.push_back(mean);
		sumSquaredErrors = 0.0;
		for(int j = 0; j < dataset.col(i).size(); j++){
			error = dataset(j, i) - mean;
			sumSquaredErrors += error * error;
		}
		//I need to check if the standard deviation is zero, if it is, 
		//I will set it to 1, so the centering is not harmed.
		if(sumSquaredErrors == 0){
			datasetStds.push_back(1.0);
		}
		else{
			datasetStds.push_back(sqrt(sumSquaredErrors/numRows));
		}
	}	
}

void centerMatrix(MatrixXi dataset, vector<float> &datasetMeans, 
	vector<float> &datasetStds, MatrixXf &centeredDataset){

	assert(datasetMeans.size() == 0);
	//I first need to obtain the mean and std of each column:
	obtainStatistics(dataset, datasetMeans, datasetStds);

	//Now I can center the matrix:
	for(int i = 0; i < dataset.rows(); i++){
		for(int j = 0; j < dataset.cols(); j++){
			centeredDataset(i, j) = (dataset(i, j) - datasetMeans[j])/datasetStds[j];
		}
	}
}

//void fillWithTopEigenVectors(int k, JacobiSVD<MatrixXf> svdResult, 
void fillWithTopEigenVectors(int k, RedSVD::RedSVD<MatrixXf> svdResult, 
	vector<vector<float> > &eigenVectors){

	assert(eigenVectors.size() == k);
	assert(eigenVectors[0].size() == svdResult.matrixV().rows());

	for(int i = 0; i < eigenVectors.size(); i++){
		for(int j = 0; j < eigenVectors[i].size(); j++){
			eigenVectors[i][j] = svdResult.matrixV()(j, i);
		}
	}
}	

void saveDecompositionInFile(vector<vector<float> > &eigenVectors,
	vector<float> &datasetMeans, vector<float> &datasetStds, int iter){

	string baseName = "svdRareEventsIter";
	stringstream sstm_fileName;
	sstm_fileName << baseName << iter + 1 << "_mean.out";
	string outputPathMean_param = sstm_fileName.str();
	sstm_fileName.str("");
	sstm_fileName << baseName << iter + 1 << "_std.out";
	string outputPathStd_param = sstm_fileName.str();


	ofstream outputStd;
	ofstream outputMean;
	outputMean.open(outputPathMean_param, ios::out);
	outputStd.open(outputPathStd_param, ios::out);
	for(int i = 0; i < datasetMeans.size(); i++){
		outputMean << datasetMeans[i] << endl;
		outputStd << datasetStds[i] << endl;
	}
	outputStd.close();
	outputMean.close();

	for(int i = 0; i < eigenVectors.size(); i++){
		sstm_fileName.str("");
		sstm_fileName << baseName << iter + 1 << "_eig" << i + 1 << ".out";
		string outputPath = sstm_fileName.str();

		ofstream output;
		output.open(outputPath, ios::out);
		for(int j = 0; j < eigenVectors[i].size(); j++){
			output << eigenVectors[i][j] << endl;
		}
		output.close();
	}

}

void saveSVDResult(MatrixXf matrix, string prefix, int iter){
	std::stringstream sstm_fileName;
	sstm_fileName << prefix << "Iter" << iter + 1 << ".csv";
	string outputPath = sstm_fileName.str();

	ofstream outFile;
	outFile.open(outputPath, ios::out);
	outFile << matrix.rows() << "," << matrix.cols() << endl;

	for(int i = 0; i < matrix.rows(); i++){
		for(int j = 0; j < matrix.cols(); j++){
			outFile << matrix(i, j) << ",";
		}
		outFile << endl;
	}
	outFile.close();
}

void reduceDimensionalityOfEvents(MatrixXi dataset, vector<float> &datasetMeans, 
	vector<float> &datasetStds, vector<vector<float> > &eigenVectors, int k, int iter){

	cout << "Running the Singular Value Decomposition in the Eigen-events\n";
	/* To run the SVD, the first thing one needs to do is to center the matrix
	 (we subtract each element by its column mean and then we divide the result
	 by its column standard deviation). Because we are going to use the eigenvectors
	 in the future, we have to store the vectors used to center the matrix. */
	MatrixXf centeredDataset(dataset.rows(), dataset.cols());
	centerMatrix(dataset, datasetMeans, datasetStds, centeredDataset);
	/* Once we centered the matrix we can run the SVD itself. We also want
	the eigenvectors, not only the eigenvalues, this is why we ask for U and V.*/
	//JacobiSVD<MatrixXf> svd(centeredDataset, ComputeFullU | ComputeFullV);
	/* I was using JacobiSVD to do the SVD but it is too slow. I am now using an
	approach based on a randomized algorithm. The algorithm I am using was
	introduced in "Finding structure with randomness: Stochastic algorithms for 
	constructing approximate matrix decompositions", N. Halko, P.G. Martinsson, 
	J. Tropp, arXiv 0909.4061"*/
	RedSVD::RedSVD<MatrixXf> svd(centeredDataset, 20);

	saveSVDResult(svd.matrixU(), "svdU", iter);
	saveSVDResult(svd.singularValues(), "svdS", iter);
	saveSVDResult(svd.matrixV(), "svdV", iter);

	//cout << svd.singularValues() << endl;
	MatrixXf m_matrixU = svd.matrixU();
	MatrixXf m_vectorS = svd.singularValues();
	MatrixXf m_matrixV = svd.matrixV();

	//MatrixXf reconstructed = m_matrixU * m_vectorS * m_matrixV.transpose();
	//cout << m_matrixU.rows() << "x" << m_matrixU.cols() << " x " << m_vectorS.rows() << "x" << m_vectorS.cols()
	//	<< " x " << m_matrixV.rows() << "x" << m_matrixV.cols() << endl;

	/* Finally, we need to obtain the top k eigenvectors (we are not concerned
	with the whole decomposition). Here I fill the matrix eigenVectors to be
	later used in the learning part of my algorithm.*/
	fillWithTopEigenVectors(k, svd, eigenVectors);
	/* We also save the obtained decomposition, in case we have to restart the
	  execution. These files should summarize everything done in this step. */
	saveDecompositionInFile(eigenVectors, datasetMeans, datasetStds, iter);
}
