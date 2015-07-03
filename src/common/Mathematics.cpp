/****************************************************************************************
** Mathematical tools that may be useful across several applications, even outside ALE
** environment. This class is meant to be static since ideally this should only implement
** functions. Adding non-static objects should be further discussed before implemented.
** 
** Author: Marlos C. Machado
***************************************************************************************/
#include <assert.h>
#include <cstdlib>
#include <vector>
namespace Mathematics
{

template<typename T>
int argmax(const T& array){
	assert(array.size() > 0);

	//We need to break ties, thus we save all  
	//indices that hold the same max value:
	std::vector<int> indices;
    auto cur_max = array[0];
	for(unsigned int i = 0; i < array.size(); i++){
        if(abs(array[i]- cur_max)<1e-10){
            indices.push_back(i);            
        }else{
            if(array[i] > cur_max){
                cur_max = array[i];
                indices.clear();
                indices.push_back(i);
            }
        }
	}
	assert(indices.size() > 0);
	//Now we randomly pick one of the best
	return indices[rand()%indices.size()];
}

template<>
inline int argmax<af::array>(const af::array& in){
    af::array indices = af::where(in == af::max<float>(in));
    unsigned chosen = rand()%indices.dims(0);
    unsigned* ret = new unsigned[1];
    ret = indices(chosen).host<unsigned>();
    return ret[0];
}

template<typename T>
void fill(T& array, float value)
{
    std::fill(array.begin(),array.end(),value);
}

template<>
inline void fill<af::array>(af::array& array, float value){
    array = af::constant(value,array.dims(),f32);
}


template<typename T>
float weighted_sparse_dotprod(const T& vec,const vector<int>& mask, float weight)
{
    return std::accumulate(mask.begin(),mask.end(), 0.0,
                           [&weight](const float& elem, const int& id){
                               return elem + weight*vec[id];
                           });
}
       

}
