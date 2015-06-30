
int main(int argc, char** argv){
	readParameters();
	for(int i = 0; i < maxNumIterations; i++){
		gatherSamplesFromRandomTrajectories();
		reduceDimensionalityOfEvents();
		learnOptionsDerivedFromEigenEvents():
	}
}