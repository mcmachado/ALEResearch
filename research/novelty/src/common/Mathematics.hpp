/********************************************************************************************
*** Mathematical tools that may be useful across several applications, even outside ALE    **
*** environment. This class is meant to be static since ideally this should only implement **
*** functions. Adding non-static objects should be further discussed before implemented.   **
***       																				   **
*** Author: Marlos C. Machado															   **
*********************************************************************************************/

#ifndef MATHEMATICS_H
#define MATHEMATICS_H

#include <vector>
#include <math.h>

class Mathematics{
	public:
		static int argmax(std::vector<float> array);
};

#endif
