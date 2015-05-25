# Makefile - need to rename this file to Makefile
#
# Author: Marlos C. Machado

ALE := ../../../MyALE/


# -O3 Optimize code (urns on all optimizations specified by -O2 and also turns on the -finline-functions, -funswitch-loops, -fpredictive-commoning, -fgcse-after-reload, -ftree-loop-vectorize, -ftree-slp-vectorize, -fvect-cost-model, -ftree-partial-pre and -fipa-cp-clone options).
# -D__USE_SDL Ensures we can use SDL to see the game screen
# -D_GNU_SOURCE=1 means the compiler will use the GNU standard of compilation, the superset of all other standards under GNU C libraries.
# -D_REENTRANT causes the compiler to use thread safe (i.e. re-entrant) versions of several functions in the C library.
FLAGS := -O3 -Wall -I$(ALE)/src -I/opt/local/include -L$(ALE) -L/opt/local/lib -D__USE_SDL -D_GNU_SOURCE=1 -D_REENTRANT `sdl-config --cflags --libs`
CXX := g++

OUT_FILE := generateSamples

# Search for library 'ale' and library 'z' when linking.
LDFLAGS := -lale -lz -lm

all: generateSamples

generateSamples: generateSamples.o Features.o RAMFeatures.o
	$(CXX) $(FLAGS) bin/generateSamples.o bin/Features.o bin/RAMFeatures.o $(LDFLAGS) -o $(OUT_FILE) 

generateSamples.o: generateSamples.cpp
	$(CXX) $(FLAGS) -c generateSamples.cpp -o bin/generateSamples.o

Features.o: ../../src/features/Features.cpp
	$(CXX) $(FLAGS) -c ../../src/features/Features.cpp -o bin/Features.o

RAMFeatures.o: ../../src/features/RAMFeatures.cpp
	$(CXX) $(FLAGS) -c ../../src/features/RAMFeatures.cpp -o bin/RAMFeatures.o

clean:
	rm -rf ${OUT_FILE} bin/*.o	

#This command needs to be executed in a osX before running the code:
#export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:../lib/ale_0_4"