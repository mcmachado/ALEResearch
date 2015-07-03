/****************************************************************************************
** Mathematical tools that may be useful across several applications, even outside ALE
** environment. This class is meant to be static since ideally this should only implement
** functions. Adding non-static objects should be further discussed before implemented.
** 
** Author: Marlos C. Machado
***************************************************************************************/
#ifndef MATH_HH
#define MATH_HH
#include <vector>
#include <math.h>
#ifdef ARRAYFIRE
#include <arrayfire.h>
#endif


namespace Mathematics{

    template<typename T> //T is a container, for example a std::vector<double>
    int argmax(const T& array);


    template<typename T>
    void fill(T& array,float value);

    template<typename T>
    float weighted_sparse_dotprod(const T& vec,const std::vector<int>& mask, float weight);

}
#include "Mathematics.cpp"
#endif
