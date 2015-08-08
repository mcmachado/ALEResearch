/****************************************************************************************
** Implementation of a variation of BASS Features, which has features to encode the 
**  relative position between tiles.
**
** REMARKS: - This implementation is basically Erik Talvitie's implementation, presented
**            in the AAAI'15 LGCVG Workshop.
**
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef REDUNDANT_BPRO_FEATURES_H
#define REDUNDANT_BPRO_FEATURES_H
#include "RedundantBPROFeatures.hpp"
#endif
#ifndef BASIC_FEATURES_H
#define BASIC_FEATURES_H
#include "BasicFeatures.hpp"
#endif

#include <set>
#include <assert.h>
//#include <boost/tuple/tuple.hpp> //TODO: I have to remove this to not have to depend on boost

RedundantBPROFeatures::RedundantBPROFeatures(Parameters *param){
    this->param = param;
    numColumns  = param->getNumColumns();
	numRows     = param->getNumRows();
	numColors   = param->getNumColors();

	if(this->param->getSubtractBackground()){
        this->background = new Background(param);
    }

	//To get the total number of features:
	//TODO: Fix this!
    numBasicFeatures = this->param->getNumColumns() * this->param->getNumRows() * this->param->getNumColors();
	numRelativeFeatures = (2 * this->param->getNumColumns() - 1) * (2 * this->param->getNumRows() - 1) 
							* this->param->getNumColors() * this->param->getNumColors();
}

RedundantBPROFeatures::~RedundantBPROFeatures(){}

int RedundantBPROFeatures::getBasicFeaturesIndices(const ALEScreen &screen, int blockWidth, int blockHeight,
	vector<vector<vector<int> > > &whichColors, vector<int>& features){
	int featureIndex = 0;
	// For each pixel block
	for (int by = 0; by < numRows; by++) {
		for (int bx = 0; bx < numColumns; bx++) {
			//vector<boost::tuple<int, int, int> > posAndColor;
			
			int xo = bx * blockWidth;
			int yo = by * blockHeight;
			vector<bool> hasColor(numColors, false);
			
			// Determine which colors are present
			for (int x = xo; x < xo + blockWidth; x++){
				for (int y = yo; y < yo + blockHeight; y++){
					unsigned char pixel = screen.get(y,x);

					if(!this->param->getSubtractBackground() || (this->background->getPixel(y, x) != pixel)){
						if(numColors == 8){ //SECAM, considering only 8 colors
							pixel = (pixel & 0xF) >> 1;
						}
						else if(numColors == 128){ //NTSC, considering 128 colors
							pixel = pixel >> 1;
						}
		  				
		  				hasColor[pixel] = true;
						//posAndColor.push_back(boost::make_tuple(x, y, pixel));
					}
				}
			}

			for(int c = 0; c < numColors; c++){
				if(hasColor[c]){
					whichColors[bx][by].push_back(c);
					features.push_back(featureIndex);
				}
				featureIndex++;
			}
		}
	}
	return featureIndex;
}

void RedundantBPROFeatures::addRelativeFeaturesIndices(const ALEScreen &screen, int featureIndex,
	vector<vector<vector<int> > > &whichColors, vector<int>& features){

	int numRowOffsets = 2*numRows - 1;
	int numColumnOffsets = 2*numColumns - 1;
	int numOffsets = numRowOffsets*numColumnOffsets;
	int numColorPairs = numColors*numColors;

	vector<bool> colorPairSeen(numColorPairs, false);

	vector<vector<bool> > colorOffsets(numOffsets, vector<bool>(numColorPairs, false));
	vector<vector<bool> > rowOffsets(numRowOffsets, vector<bool>(numColorPairs, false));
	vector<vector<bool> > columnOffsets(numColumnOffsets, vector<bool>(numColorPairs, false));
	vector<vector<bool> > quadrantOffsets(8, vector<bool>(numColorPairs, false));
	for(int bx = numColumns; bx--;){
		for(int by = numRows; by--;){
			for(int offX = numColumns; offX--;){
				for(int offY = numRows; offY--;){
					int xOff = offX - bx + numColumns - 1;
					int yOff = offY - by + numRows - 1;
					int offset = yOff*(2*numRows - 1) + xOff;

					int numBColors = whichColors[bx][by].size();
					int numOffColors = whichColors[offX][offY].size();
					for(int c = numBColors; c--;){
						int bColor = whichColors[bx][by][c];
						for(int offC = numOffColors; offC--;){
							int colorPair = bColor*numColors + whichColors[offX][offY][offC];
							colorPairSeen[colorPair] = true;

							colorOffsets[offset][colorPair] = true;
							rowOffsets[yOff][colorPair] = true;
							columnOffsets[xOff][colorPair] = true;
							if(offX - bx > 0){
								quadrantOffsets[0][colorPair] = true;
								if(offY - by > 0){
			     					quadrantOffsets[1][colorPair] = true;
			   					}
								else if(offY - by < 0){
									quadrantOffsets[2][colorPair] = true;
								}
							}
							else if(offX - bx < 0){
								quadrantOffsets[3][colorPair] = true;
								if(offY - by > 0){
									quadrantOffsets[4][colorPair] = true;
								}
								else if(offY - by < 0){
									quadrantOffsets[5][colorPair] = true;
								}
							}
							if(offY - by > 0){
								quadrantOffsets[6][colorPair] = true;
							}
							else if(offY - by < 0){
								quadrantOffsets[7][colorPair] = true;
							}
						}
					}
				}
			}
		}
	}

	vector<int> seenColorPairs;
	for(int i = 0; i < numColorPairs; i++){
		if(colorPairSeen[i]){
			seenColorPairs.push_back(i);
		}
	}

	for(unsigned o = numOffsets; o--;){
		for(unsigned i = 0; i < seenColorPairs.size(); i++){
			int colorPair = seenColorPairs[i];
			if(colorOffsets[o][colorPair]){
				features.push_back(featureIndex + colorPair);
			}
		}
		featureIndex += numColorPairs;
	}
}

void RedundantBPROFeatures::getActiveFeaturesIndices(const ALEScreen &screen, const ALERAM &ram, vector<int>& features){
	int screenWidth = screen.width();
	int screenHeight = screen.height();
	int blockWidth = screenWidth / numColumns;
	int blockHeight = screenHeight / numRows;

	assert(features.size() == 0); //If the vector is not empty this can be a mess
	vector<vector<vector<int> > > whichColors(numColumns, vector<vector<int> >(numRows));

    //Before generating features we must check whether we can subtract the background:
    if(this->param->getSubtractBackground()){
        unsigned int sizeBackground = this->background->getWidth() * this->background->getHeight();
        assert(sizeBackground == screen.width()*screen.height());
    }

    //We first get the Basic features, keeping track of the next featureIndex vector:
    //We don't just use the Basic implementation because we need the whichColors information
	int featureIndex = getBasicFeaturesIndices(screen, blockWidth, blockHeight, whichColors, features);
	addRelativeFeaturesIndices(screen, featureIndex, whichColors, features);

	//Bias
	features.push_back(featureIndex);
}

int RedundantBPROFeatures::getNumberOfFeatures(){
    return numBasicFeatures + numRelativeFeatures + 1;
}