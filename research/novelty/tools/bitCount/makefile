# Makefile - need to rename this file to Makefile
#
# Author: Marlos C. Machado

ALE := ../../../../../Arcade-Learning-Environment/


CXX := g++ -std=c++11
FLAGS := -O3 -I$(ALE)/src -D__USE_SDL `sdl-config --cflags --libs`
LDFLAGS := -lale -lz -lm -lSDL -lSDL_gfx -lSDL_image

OUT_FILE1 := humanPlayer
OUT_FILE2 := randomPlayer
OUT_FILE3 := optionsPlayer

all: humanPlayer randomPlayer optionsPlayer

humanPlayer: bin/human.o bin/HumanPlayer.o bin/RAMFeatures.o
	$(CXX) $(FLAGS) -L$(ALE) bin/human.o bin/HumanPlayer.o bin/RAMFeatures.o $(LDFLAGS) -o $(OUT_FILE1)

randomPlayer: bin/random.o bin/HumanPlayer.o bin/RAMFeatures.o
	$(CXX) $(FLAGS) -L$(ALE) bin/random.o bin/HumanPlayer.o bin/RAMFeatures.o $(LDFLAGS) -o $(OUT_FILE2)

optionsPlayer: bin/options.o bin/RAMFeatures.o bin/BPROFeatures.o bin/Background.o
	$(CXX) $(FLAGS) -L$(ALE) bin/options.o bin/RAMFeatures.o bin/BPROFeatures.o bin/Background.o $(LDFLAGS) -o $(OUT_FILE3)

bin/random.o: random.cpp
	$(CXX) $(FLAGS) -c random.cpp -o bin/random.o

bin/human.o: human.cpp
	$(CXX) $(FLAGS) -c human.cpp -o bin/human.o

bin/options.o: options.cpp
	$(CXX) $(FLAGS) -c options.cpp -o bin/options.o

bin/HumanPlayer.o: HumanPlayer.cpp
	$(CXX) $(FLAGS) -c HumanPlayer.cpp -o bin/HumanPlayer.o

bin/RAMFeatures.o: RAMFeatures.cpp
	$(CXX) $(FLAGS) -c RAMFeatures.cpp -o bin/RAMFeatures.o

bin/BPROFeatures.o: BPROFeatures.cpp
	$(CXX) $(FLAGS) -c BPROFeatures.cpp -o bin/BPROFeatures.o

bin/Background.o: Background.cpp
	$(CXX) $(FLAGS) -c Background.cpp -o bin/Background.o

clean:
	rm -rf ${OUT_FILE} bin/*.o	


#This command needs to be executed in a osX before running the code:
#export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:../lib/ale_0_4"