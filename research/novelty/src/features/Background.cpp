/****************************************************************************************
** This class is used to store the background, which may be removed from features.
** 
** Author: Marlos C. Machado
***************************************************************************************/

#include <fstream>
#include <stdlib.h>
#include "Background.hpp"

Background::Background(std::string gameName){
	std::string line;
	std::string token;
	std::string delimiter = ",";
	int row = 0;
	int col = 0;
	int ratio = 2;

	std::string pathToBackground = "../../../../data/backgrounds/" + gameName + ".bg";
	//Open background file
	std::ifstream backgroundFile(pathToBackground.c_str());

	if (backgroundFile.is_open()){
		//First read the matrix dimensions and allocate background
		getline(backgroundFile, line);
		
		size_t pos = 0;
		//I assume the first line is the height x width
		
		pos = line.find(delimiter);
		this->width = atoi(line.substr(0, pos).c_str());
		line.erase(0, pos + delimiter.length());
		this->height = atoi(line.c_str());

		//Dynamically allocate the background matrix
		for(int i = 0; i < height; i++){
			background.push_back(std::vector<int>(width, 0));
		}

		//Read file line by line, adding parsed elements to matrix:
		while(getline(backgroundFile, line)){
			if(line.length() > 0){
				col = 0;
				while ((pos = line.find(delimiter)) != std::string::npos) {
			    	this->background[row][col] = atoi(line.substr(0, pos).c_str());
			    	line.erase(0, pos + 1);
			    	col++;
				}
				this->background[row][col] = atoi(line.c_str());
				row++;
			}
		}
		backgroundFile.close();
	}
}

int Background::getPixel(int x, int y){
	return this->background[x][y];
}

int Background::getWidth(){
	return this->width;
}

int Background::getHeight(){
	return this->height;
}

Background::~Background(){}
