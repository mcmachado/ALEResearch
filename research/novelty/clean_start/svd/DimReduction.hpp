/* Author: Marlos C. Machado */

#include <vector>
#include <Eigen/SVD>
#include <Eigen/Dense>

void centerMatrix(Eigen::MatrixXi dataset, Eigen::MatrixXf &centeredDataset);

void fillWithTopEigenVectors(int k, std::vector<std::vector<float> > &eigenVectors);

void fillCenteringVector(std::vector<float> &centeringVector);

Eigen::JacobiSVD<Eigen::MatrixXf> reduceDimensionalityOfEvents(Eigen::MatrixXi dataset, 
	std::vector<float> &centeringVector, std::vector<std::vector<float> > &eigenVectors, int k);