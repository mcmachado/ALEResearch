/****************************************************************************************
** This class is used to store the background, which may be subtracted from screen to
** generate features. This approach was suggested in the JAIR paper and drastically
** reduces the number of features in the problem.
** 
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef BACKGROUND_H
#define BACKGROUND_H
#include "../input/Parameters.hpp"

class Background{
	private:
		int width;
		int height;
		int down_width;
		int down_height;
		std::vector<std::vector<int> > background;

		/**
		* Constructor, private so no one calls it without the proper information.
		*/
		Background();
		
	public:
		/**
		* Constructor to be used.
		*/
		Background(Parameters *param);

		/**
		* Destructor used to delete the background, which is allocated dynamically
		*/
		~Background();

		/**
		* Method used to retrieve a pixel from the background.
		*/
		int getPixel(int x, int y);

		/**
		* Return the background screen width
		*/
		int getWidth();

		/**
		* Return the background screen height
		*/
		int getHeight();
};

#endif
