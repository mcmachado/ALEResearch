# Makefile - need to rename this file to Makefile
#
# Author: Marlos C. Machado

ALE := ../../../../../Arcade-Learning-Environment/


CXX := g++ -std=c++11
FLAGS := -O3 -I$(ALE)/src -D__USE_SDL `sdl-config --cflags --libs`
LDFLAGS := -lale -lz -lm -lSDL -lSDL_gfx -lSDL_image

OUT_FILE2 := randomPlayer
OUT_FILE3 := optionsPlayer

all: randomPlayer optionsPlayer

randomPlayer: bin/generateHistogramsFromBytesRandom.o
	$(CXX) $(FLAGS) -L$(ALE) bin/generateHistogramsFromBytesRandom.o $(LDFLAGS) -o $(OUT_FILE2)

optionsPlayer: bin/generateHistogramsFromBytesOptions.o bin/BPROFeatures.o bin/Background.o
	$(CXX) $(FLAGS) -L$(ALE) bin/generateHistogramsFromBytesOptions.o bin/BPROFeatures.o bin/Background.o $(LDFLAGS) -o $(OUT_FILE3)

bin/generateHistogramsFromBytesRandom.o: generateHistogramsFromBytesRandom.cpp
	$(CXX) $(FLAGS) -c generateHistogramsFromBytesRandom.cpp -o bin/generateHistogramsFromBytesRandom.o

bin/generateHistogramsFromBytesOptions.o: generateHistogramsFromBytesOptions.cpp
	$(CXX) $(FLAGS) -c generateHistogramsFromBytesOptions.cpp -o bin/generateHistogramsFromBytesOptions.o

bin/BPROFeatures.o: BPROFeatures.cpp
	$(CXX) $(FLAGS) -c BPROFeatures.cpp -o bin/BPROFeatures.o

bin/Background.o: Background.cpp
	$(CXX) $(FLAGS) -c Background.cpp -o bin/Background.o

clean:
	rm -rf ${OUT_FILE} bin/*.o	


#This command needs to be executed in a osX before running the code:
#export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:../lib/ale_0_4"