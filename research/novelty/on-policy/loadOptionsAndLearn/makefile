# Makefile
#
# Author: Marlos C. Machado

ALE := ../../../../../MyALE/

# Set this to 1 to enable SDL and display_screen
USE_SDL     := 1

# -O3 Optimize code (urns on all optimizations specified by -O2 and also turns on the -finline-functions, -funswitch-loops, -fpredictive-commoning, -fgcse-after-reload, -ftree-loop-vectorize, -ftree-slp-vectorize, -fvect-cost-model, -ftree-partial-pre and -fipa-cp-clone options).
# -D__USE_SDL Ensures we can use SDL to see the game screen
# -D_GNU_SOURCE=1 means the compiler will use the GNU standard of compilation, the superset of all other standards under GNU C libraries.
# -D_REENTRANT causes the compiler to use thread safe (i.e. re-entrant) versions of several functions in the C library.
FLAGS := -O3 -I$(ALE)/src -I/opt/local/include -L$(ALE) -D_GNU_SOURCE=1 -D_REENTRANT -D__STDC_CONSTANT_MACROS
CXX := g++ -std=c++11
OUT_FILE1 := sarsaLoadWeightsProxyOption
OUT_FILE2 := sarsaMaxScore
OUT_FILE3 := sarsaExpReplay
# Search for library 'ale' and library 'z' when linking.
LDFLAGS := -lale -lz -lm

ifeq ($(strip $(USE_SDL)), 1)
  FLAGS +=  -D__USE_SDL `sdl-config --cflags --libs`
  LDFLAGS += -lSDL -lSDL_gfx -lSDL_image 
endif

all: sarsaLoadWeightsProxyOption sarsaMaxScore sarsaExpReplay

sarsaLoadWeightsProxyOption: sarsaLoadWeightsProxyOption.o     Mathematics.o     Timer.o     ParametersLoadingWeights.o     Features.o     Background.o     BPROFeatures.o     RAMFeatures.o     RLLearnerExtended.o     OptionSarsaExtended.o
	$(CXX) $(FLAGS)      bin/sarsaLoadWeightsProxyOption.o bin/Mathematics.o bin/Timer.o bin/ParametersLoadingWeights.o bin/Features.o bin/Background.o bin/BPROFeatures.o bin/RAMFeatures.o bin/RLLearnerExtended.o bin/OptionSarsaExtended.o $(LDFLAGS) -o $(OUT_FILE1)

sarsaMaxScore:               sarsaLoadWeightsMaxScore.o     Mathematics.o     Timer.o     ParametersLoadingWeights.o     Features.o     Background.o     BPROFeatures.o     RAMFeatures.o     RLLearnerMaxScore.o     SarsaWithOptionsMaxScore.o
	$(CXX) $(FLAGS)      bin/sarsaLoadWeightsMaxScore.o bin/Mathematics.o bin/Timer.o bin/ParametersLoadingWeights.o bin/Features.o bin/Background.o bin/BPROFeatures.o bin/RAMFeatures.o bin/RLLearnerMaxScore.o bin/SarsaWithOptionsMaxScore.o $(LDFLAGS) -o $(OUT_FILE2)

sarsaExpReplay:              sarsaExpReplayMaxScore.o     Mathematics.o     Timer.o     ParametersLoadingWeights.o     Features.o     Background.o     BPROFeatures.o     RAMFeatures.o     RLLearnerExpReplay.o     SarsaLearnerExpReplay.o
	$(CXX) $(FLAGS)      bin/sarsaExpReplayMaxScore.o bin/Mathematics.o bin/Timer.o bin/ParametersLoadingWeights.o bin/Features.o bin/Background.o bin/BPROFeatures.o bin/RAMFeatures.o bin/RLLearnerExpReplay.o bin/SarsaLearnerExpReplay.o $(LDFLAGS) -o $(OUT_FILE3)

sarsaLoadWeightsProxyOption.o: sarsaLoadWeightsProxyOption.cpp
	$(CXX) $(FLAGS) -c sarsaLoadWeightsProxyOption.cpp -o bin/sarsaLoadWeightsProxyOption.o

sarsaLoadWeightsMaxScore.o: sarsaLoadWeightsMaxScore.cpp
	$(CXX) $(FLAGS) -c sarsaLoadWeightsMaxScore.cpp -o bin/sarsaLoadWeightsMaxScore.o

sarsaExpReplayMaxScore.o: sarsaExpReplayMaxScore.cpp
	$(CXX) $(FLAGS) -c sarsaExpReplayMaxScore.cpp -o bin/sarsaExpReplayMaxScore.o

Mathematics.o: ../../../../src/common/Mathematics.cpp
	$(CXX) $(FLAGS) -c ../../../../src/common/Mathematics.cpp -o bin/Mathematics.o

Timer.o: ../../../../src/common/Timer.cpp
	$(CXX) $(FLAGS) -c ../../../../src/common/Timer.cpp -o bin/Timer.o

ParametersLoadingWeights.o: common/ParametersLoadingWeights.cpp
	$(CXX) $(FLAGS) -c      common/ParametersLoadingWeights.cpp -o bin/ParametersLoadingWeights.o

Features.o: ../../../../src/features/Features.cpp
	$(CXX) $(FLAGS) -c ../../../../src/features/Features.cpp -o bin/Features.o

Background.o: ../../../../src/features/Background.cpp
	$(CXX) $(FLAGS) -c ../../../../src/features/Background.cpp -o bin/Background.o	
	
BPROFeatures.o: ../../../../src/features/BPROFeatures_Old.cpp
	$(CXX) $(FLAGS) -c ../../../../src/features/BPROFeatures_Old.cpp -o bin/BPROFeatures.o	

RAMFeatures.o: ../../../../src/features/RAMFeatures.cpp
	$(CXX) $(FLAGS) -c ../../../../src/features/RAMFeatures.cpp -o bin/RAMFeatures.o		

RLLearnerMaxScore.o: control/RLLearnerMaxScore.cpp
	$(CXX) $(FLAGS) -c control/RLLearnerMaxScore.cpp -o bin/RLLearnerMaxScore.o	

RLLearnerExtended.o: control/RLLearnerExtended.cpp
	$(CXX) $(FLAGS) -c control/RLLearnerExtended.cpp -o bin/RLLearnerExtended.o	

OptionSarsaExtended.o: control/OptionSarsaExtended.cpp
	$(CXX) $(FLAGS) -c control/OptionSarsaExtended.cpp -o bin/OptionSarsaExtended.o

SarsaWithOptionsMaxScore.o: control/SarsaWithOptionsMaxScore.cpp
	$(CXX) $(FLAGS) -c control/SarsaWithOptionsMaxScore.cpp -o bin/SarsaWithOptionsMaxScore.o

SarsaLearnerExpReplay.o: control/SarsaLearnerExpReplay.cpp
	$(CXX) $(FLAGS) -c control/SarsaLearnerExpReplay.cpp -o bin/SarsaLearnerExpReplay.o

RLLearnerExpReplay.o: control/RLLearnerExpReplay.cpp
	$(CXX) $(FLAGS) -c control/RLLearnerExpReplay.cpp -o bin/RLLearnerExpReplay.o

clean:
	rm -rf ${OUT_FILE} bin/*.o	


#This command needs to be executed in a osX before running the code:
#export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:../../../../MyALE/"
