/****************************************************************************************
** This class is used to store the background, which may be subtracted from screen to
** generate features. This approach was suggested in the JAIR paper and drastically
** reduces the number of features in the problem.
** 
** Author: Marlos C. Machado
***************************************************************************************/

#include <vector>

class Background{
	private:
		int width;
		int height;
		int down_width;
		int down_height;
		std::vector<std::vector<int> > background;

	public:
		/**
		* Constructor to be used.
		* @param Parameters param contains the path to the background file
		*/
		Background();
		/**
		* Destructor used to delete the background, which is allocated dynamically
		*/
		~Background();
		/**
		* Method used to retrieve a pixel from the background.
		* 
		* TODO: Make it return an unsigned char, it is more efficient.
		* 
		* @param int x coordinate
		* @param int y coordinate
		*
		* @return pixel value in the coordinate (x, y)
		*/
		int getPixel(int x, int y);
		/**
		* @return int background screen width
		*/
		int getWidth();
		/**
		* @return int background screen height
		*/
		int getHeight();
};
