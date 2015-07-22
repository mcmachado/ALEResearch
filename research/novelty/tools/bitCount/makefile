# Makefile - need to rename this file to Makefile
#
# Author: Marlos C. Machado

ALE := ../../../../../Arcade-Learning-Environment/


FLAGS := -O3 -I$(ALE)/src -D__USE_SDL `sdl-config --cflags --libs`
CXX := g++ -std=c++11
OUT_FILE := humanPlayer
LDFLAGS := -lale -lz -lm -lSDL -lSDL_gfx -lSDL_image

all: humanPlayer

humanPlayer: bin/main.o bin/HumanPlayer.o bin/RAMFeatures.o
	$(CXX) $(FLAGS) -L$(ALE) bin/main.o bin/HumanPlayer.o bin/RAMFeatures.o $(LDFLAGS) -o $(OUT_FILE)

bin/main.o: main.cpp
	$(CXX) $(FLAGS) -c main.cpp -o bin/main.o

bin/HumanPlayer.o: HumanPlayer.cpp
	$(CXX) $(FLAGS) -c HumanPlayer.cpp -o bin/HumanPlayer.o

bin/RAMFeatures.o: RAMFeatures.cpp
	$(CXX) $(FLAGS) -c RAMFeatures.cpp -o bin/RAMFeatures.o

clean:
	rm -rf ${OUT_FILE} bin/*.o	


#This command needs to be executed in a osX before running the code:
#export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:../lib/ale_0_4"