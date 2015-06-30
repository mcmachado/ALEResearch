# Makefile
#
# Author: Marlos C. Machado

ALE := ../../../../MyALE/

# Set this to 1 to enable SDL and display_screen
USE_SDL     := 1

# -O3 Optimize code (urns on all optimizations specified by -O2 and also turns on the -finline-functions, -funswitch-loops, -fpredictive-commoning, -fgcse-after-reload, -ftree-loop-vectorize, -ftree-slp-vectorize, -fvect-cost-model, -ftree-partial-pre and -fipa-cp-clone options).
# -D__USE_SDL Ensures we can use SDL to see the game screen
# -D_GNU_SOURCE=1 means the compiler will use the GNU standard of compilation, the superset of all other standards under GNU C libraries.
# -D_REENTRANT causes the compiler to use thread safe (i.e. re-entrant) versions of several functions in the C library.
FLAGS := -O3 -I$(ALE)/src -I/opt/local/include -L$(ALE) -D_GNU_SOURCE=1 -D_REENTRANT -D__STDC_CONSTANT_MACROS
CXX := g++
OUT_FILE1 := sarsaProxyOption
OUT_FILE2 := loadAndLearn
OUT_FILE3 := sarsaLoadWeightsProxyOption
# Search for library 'ale' and library 'z' when linking.
LDFLAGS := -lale -lz -lm

ifeq ($(strip $(USE_SDL)), 1)
  FLAGS +=  -D__USE_SDL `sdl-config --cflags --libs`
  LDFLAGS += -lSDL -lSDL_gfx -lSDL_image 
endif

all: sarsaProxyOption loadAndLearn

sarsaProxyOption:       sarsaProxyOption.o     Mathematics.o     Timer.o     Parameters.o     Features.o     Background.o     BPROFeatures.o     RAMFeatures.o     RLLearner.o     OptionSarsa.o
	$(CXX) $(FLAGS) bin/sarsaProxyOption.o bin/Mathematics.o bin/Timer.o bin/Parameters.o bin/Features.o bin/Background.o bin/BPROFeatures.o bin/RAMFeatures.o bin/RLLearner.o bin/OptionSarsa.o $(LDFLAGS) -o $(OUT_FILE1) 

loadAndLearn:           loadAndLearn.o     Mathematics.o     Timer.o     Features.o     BackgroundLocal.o     BPROFeaturesLocal.o     RAMFeatures.o 
	$(CXX) $(FLAGS) bin/loadAndLearn.o bin/Mathematics.o bin/Timer.o bin/Features.o bin/BackgroundLocal.o bin/BPROFeaturesLocal.o bin/RAMFeatures.o  $(LDFLAGS) -o $(OUT_FILE2) 

loadAndLearn.o: loadAndLearn.cpp
	$(CXX) $(FLAGS) -c loadAndLearn.cpp -o bin/loadAndLearn.o

sarsaProxyOption.o: sarsaProxyOption.cpp
	$(CXX) $(FLAGS) -c sarsaProxyOption.cpp -o bin/sarsaProxyOption.o

Mathematics.o: ../../../src/common/Mathematics.cpp
	$(CXX) $(FLAGS) -c ../../../src/common/Mathematics.cpp -o bin/Mathematics.o

Timer.o: ../../../src/common/Timer.cpp
	$(CXX) $(FLAGS) -c ../../../src/common/Timer.cpp -o bin/Timer.o

Parameters.o: common/Parameters.cpp
	$(CXX) $(FLAGS) -c common/Parameters.cpp -o bin/Parameters.o

Features.o: ../../../src/features/Features.cpp
	$(CXX) $(FLAGS) -c ../../../src/features/Features.cpp -o bin/Features.o

Background.o: ../../../src/features/Background.cpp
	$(CXX) $(FLAGS) -c ../../../src/features/Background.cpp -o bin/Background.o	

BackgroundLocal.o: features/Background.cpp
	$(CXX) $(FLAGS) -c features/Background.cpp -o bin/BackgroundLocal.o		
	
BPROFeatures.o: ../../../src/features/BPROFeatures.cpp
	$(CXX) $(FLAGS) -c ../../../src/features/BPROFeatures.cpp -o bin/BPROFeatures.o	

BPROFeaturesLocal.o: features/BPROFeatures.cpp
	$(CXX) $(FLAGS) -c features/BPROFeatures.cpp -o bin/BPROFeaturesLocal.o	

RAMFeatures.o: ../../../src/features/RAMFeatures.cpp
	$(CXX) $(FLAGS) -c ../../../src/features/RAMFeatures.cpp -o bin/RAMFeatures.o		

RLLearner.o: control/RLLearner.cpp
	$(CXX) $(FLAGS) -c control/RLLearner.cpp -o bin/RLLearner.o

OptionSarsa.o: control/OptionSarsa.cpp
	$(CXX) $(FLAGS) -c control/OptionSarsa.cpp -o bin/OptionSarsa.o

clean:
	rm -rf ${OUT_FILE} bin/*.o	


#This command needs to be executed in a osX before running the code:
#export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:../../../../MyALE/"
