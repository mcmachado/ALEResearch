/* Author: Marlos C. Machado */


#ifndef AGENT_H
#define AGENT_H

#include <vector>

class Agent{
	public:
		int numberOfAvailActions, numberOfPrimitiveActions;
		std::vector<std::vector<std::vector<float> > > w;  //Theta, weights vector
	private:
};

#endif