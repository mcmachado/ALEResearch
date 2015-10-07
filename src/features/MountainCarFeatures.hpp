/**
 * @file   GridFeatures.hpp
 * @author Nicolas Carion
 * @date   Wed Jun  3 08:56:48 2015
 * 
 * @brief  This file provides feature classes for the Mountain Car Environment.
 * The tiling code is borrowed to Rich Sutton.
 * 
 * 
 */
#ifndef MCFEAT_H
#define MCFEAT_H

#include "../environments/mountain_car/MountainCarEnvironment.hpp"
#include <vector>

#define MAX_NUM_VARS 20

int hash_coordinates(int *coordinates, int num_indices, int memory_size);

void GetTiles(
	int tiles[],               // provided array contains returned tiles (tile indices)
	int num_tilings,           // number of tile indices to be returned in tiles       
	float variables[],         // array of variables
    int num_variables,         // number of variables
    int memory_size,           // total number of possible tiles (memory size)
    int hash1 = -1,            // change these from -1 to get a different hashing
    int hash2 = -1,
    int hash3 = -1){
		int i,j;
		int qstate[MAX_NUM_VARS];
		int base[MAX_NUM_VARS];
		int coordinates[MAX_NUM_VARS + 4];   /* one interval number per rel dimension */
		int num_coordinates;
		
		if (hash1 == -1)
			num_coordinates = num_variables + 1;       // no additional hashing corrdinates
		else if (hash2 == -1) {
			num_coordinates = num_variables + 2;       // one additional hashing coordinates
			coordinates[num_variables+1] = hash1;
	}
	else if (hash3 == -1) {
		num_coordinates = num_variables + 3;       // two additional hashing coordinates
		coordinates[num_variables+1] = hash1;
		coordinates[num_variables+2] = hash2;
    }
    else {
		num_coordinates = num_variables + 4;       // three additional hashing coordinates
		coordinates[num_variables+1] = hash1;
		coordinates[num_variables+2] = hash2;
		coordinates[num_variables+3] = hash3;
    }
    
	/* quantize state to integers (henceforth, tile widths == num_tilings) */
    for (i = 0; i < num_variables; i++) {
    	qstate[i] = (int) floor(variables[i] * num_tilings);
    	base[i] = 0;
    }
    
    /*compute the tile numbers */
    for (j = 0; j < num_tilings; j++) {
    
		/* loop over each relevant dimension */
		for (i = 0; i < num_variables; i++) {
		
    		/* find coordinates of activated tile in tiling space */
			if (qstate[i] >= base[i])
				coordinates[i] = qstate[i] - ((qstate[i] - base[i]) % num_tilings);
			else
				coordinates[i] = qstate[i]+1 + ((base[i] - qstate[i] - 1) % num_tilings) - num_tilings;
				        
			/* compute displacement of next tiling in quantized space */
			base[i] += 1 + (2 * i);
		}
		/* add additional indices for tiling and hashing_set so they hash differently */
		coordinates[i++] = j;
		
		tiles[j] = hash_coordinates(coordinates, num_coordinates, memory_size);
	}
	return;
}

			
/* hash_coordinates
   Takes an array of integer coordinates and returns the corresponding tile after hashing 
*/
int hash_coordinates(int *coordinates, int num_indices, int memory_size){
	static int first_call = 1;
	static unsigned int rndseq[2048];
	int i,k;
	long index;
	long sum = 0;
	
	/* if first call to hashing, initialize table of random numbers */
    if (first_call) {
		for (k = 0; k < 2048; k++) {
			rndseq[k] = 0;
			for (i=0; i < sizeof(int); ++i)
	    		rndseq[k] = (rndseq[k] << 8) | (rand() & 0xff);    
		}
        first_call = 0;
    }

	for (i = 0; i < num_indices; i++){
		/* add random table offset for this dimension and wrap around */
		index = coordinates[i];
		index += (449 * i);
		index %= 2048;
		while (index < 0) index += 2048;
			
		/* add selected random number to sum */
		sum += (long)rndseq[(int)index];
	}
	index = (int)(sum % memory_size);
	while (index < 0) index += memory_size;
	
	return(index);
}


class MountainCarFeatures;

class MountainCarFeatures{
public:
    typedef bool FeatureType;
    
    MountainCarFeatures(){}

    void getCompleteFeatureVector(MountainCarEnvironment<MountainCarFeatures>* env,std::vector<bool>& features){
        features.clear();
        features.resize(10000,false);
        int *tiles = new int[10000];
        float vars[2] = {env->getPos(),env->getVel()};
        GetTiles(tiles,10,vars,2,10000);
        for(int i=0;i<10000;i++){
            features[tiles[i]] = true;
            std::cout<<tiles[i]<<std::endl;
        }
        delete[] tiles;
    }

    int getNumberOfFeatures(MountainCarEnvironment<MountainCarFeatures>*){
        return 10000;
    }

    void getActiveFeaturesIndices(MountainCarEnvironment<MountainCarFeatures>* env,std::vector<std::pair<int,bool>>& active){
        int *tiles = new int[10000];
        for(int i=0;i<10000;i++){
            tiles[i] = 0;
        }
        float vars[2] = {env->getPos(),env->getVel()};
        GetTiles(tiles,10,vars,2,10000);
        active.clear();
        active.push_back({0,true});
        for(int i=0;i<10000;i++){
            assert(tiles[i]<10000&&tiles[i]>=0);
            if(tiles[i] != 0)
                active.push_back({tiles[i],true});
        }
        delete[] tiles;
    }
};

#endif