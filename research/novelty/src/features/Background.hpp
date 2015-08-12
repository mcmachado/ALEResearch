/****************************************************************************************
*** This class is used to store the background, which may be subtracted from screen to **
*** generate features. This approach was suggested in the JAIR paper and drastically   **
*** reduces the number of features in the problem.									   **
*** 																				   **
*** Author: Marlos C. Machado														   **
*****************************************************************************************/

#ifndef BACKGROUND_H
#define BACKGROUND_H

#include <vector>

class Background{
	private:
		int width;
		int height;
		int down_width;
		int down_height;
		std::vector<std::vector<int> > background;

	public:
		Background(std::string gameName);
		~Background();
		
		int getPixel(int x, int y);
		int getWidth();
		int getHeight();
};
#endif
