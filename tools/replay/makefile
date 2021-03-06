# Makefile
# Author: Marlos C. Machado

ALE := ../../../MyALE/

# Set this to 1 to enable SDL and display_screen
USE_SDL     := 1

# -O3 Optimize code (urns on all optimizations specified by -O2 and also turns on the -finline-functions, -funswitch-loops, -fpredictive-commoning, -fgcse-after-reload, -ftree-loop-vectorize, -ftree-slp-vectorize, -fvect-cost-model, -ftree-partial-pre and -fipa-cp-clone options).
# -D__USE_SDL Ensures we can use SDL to see the game screen
# -D_GNU_SOURCE=1 means the compiler will use the GNU standard of compilation, the superset of all other standards under GNU C libraries.
# -D_REENTRANT causes the compiler to use thread safe (i.e. re-entrant) versions of several functions in the C library.
FLAGS := -O3 -I$(ALE)/src -I/opt/local/include -L$(ALE) -D_GNU_SOURCE=1 -D_REENTRANT
CXX := g++
OUT_FILE1 := replay
OUT_FILE2 := intrRewardOptionsFreq
OUT_FILE3 := loadOptionsAndReplay

# Search for library 'ale' and library 'z' when linking.
LDFLAGS := -lale -lz -lm

ifeq ($(strip $(USE_SDL)), 1)
  FLAGS +=  -D__USE_SDL `sdl-config --cflags --libs`
  LDFLAGS += -lSDL -lSDL_gfx -lSDL_image 
endif

all: replay intrRewardOptionsFreq loadOptionsAndReplay

replay:                 main.o     BPROFeatures.o     Background.o
	$(CXX) $(FLAGS) bin/main.o bin/BPROFeatures.o bin/Background.o $(LDFLAGS) -o $(OUT_FILE1) 

intrRewardOptionsFreq:  intrRewardOptionsFreq.o     BPROFeatures.o     RAMFeatures.o     Background.o
	$(CXX) $(FLAGS) bin/intrRewardOptionsFreq.o bin/BPROFeatures.o bin/RAMFeatures.o bin/Background.o $(LDFLAGS) -o $(OUT_FILE2) 

loadOptionsAndReplay:   loadOptionsAndReplay.o     BPROFeatures.o     Background.o
	$(CXX) $(FLAGS) bin/loadOptionsAndReplay.o bin/BPROFeatures.o bin/Background.o $(LDFLAGS) -o $(OUT_FILE3) 

loadOptionsAndReplay.o: loadOptionsAndReplay.cpp
	$(CXX) $(FLAGS) -c loadOptionsAndReplay.cpp -o bin/loadOptionsAndReplay.o

intrRewardOptionsFreq.o: intrRewardOptionsFreq.cpp
	$(CXX) $(FLAGS) -c intrRewardOptionsFreq.cpp -o bin/intrRewardOptionsFreq.o

main.o: main.cpp
	$(CXX) $(FLAGS) -c main.cpp -o bin/main.o

BPROFeatures.o: BPROFeatures.cpp
	$(CXX) $(FLAGS) -c BPROFeatures.cpp -o bin/BPROFeatures.o

RAMFeatures.o: RAMFeatures.cpp
	$(CXX) $(FLAGS) -c RAMFeatures.cpp -o bin/RAMFeatures.o

Background.o: Background.cpp
	$(CXX) $(FLAGS) -c Background.cpp -o bin/Background.o

clean:
	rm -rf ${OUT_FILE} bin/*.o	

#This command needs to be executed in a osX before running the code:
#export DYLD_LIBRARY_PATH="../../../MyALE/"
