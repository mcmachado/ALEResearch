/* Author: Marlos C. Machado */

#include <vector>
#include <RedSVD.h>
#include <Eigen/SVD>
#include <Eigen/Dense>

/* To center a matrix we obtain the mean mu and the standard deviation sig of each of its
  columns. Then we iterate over each entry A[i,j] subtracting mu and then dividing by sig*/
void centerMatrix(Eigen::MatrixXi dataset, std::vector<float> &datasetMeans, 
	std::vector<float> &datasetStds, Eigen::MatrixXf &centeredDataset);

//void fillWithTopEigenVectors(int k, Eigen::JacobiSVD<Eigen::MatrixXf> svdResult, 
void fillWithTopEigenVectors(int k, RedSVD::RedSVD<Eigen::MatrixXf> svdResult, 
	std::vector<std::vector<float> > &eigenVectors);

void obtainStatistics(Eigen::MatrixXi dataset, std::vector<float> &datasetMeans, 
	std::vector<float> &datasetStds);

void saveDecompositionInFile(std::vector<std::vector<float> > &eigenVectors,
	std::vector<float> &datasetMeans, std::vector<float> &datasetStds, int iter);

void reduceDimensionalityOfEvents(Eigen::MatrixXi dataset, std::vector<float> &datasetMeans, 
	std::vector<float> &datasetStds,  std::vector<std::vector<float> > &eigenVectors, int k, int iter);
