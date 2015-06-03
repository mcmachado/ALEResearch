# Makefile
#
# Author: Marlos C. Machado, Nicolas Carion

CXX := g++

ALE := ../../Arcade-Learning-Environment

# Set this to 1 to enable SDL and display_screen
USE_SDL     := 1

# -O3 Optimize code (turns on all optimizations specified by -O2 and also turns on the -finline-functions, -funswitch-loops, -fpredictive-commoning, -fgcse-after-reload, -ftree-loop-vectorize, -ftree-slp-vectorize, -fvect-cost-model, -ftree-partial-pre and -fipa-cp-clone options).
# -D__USE_SDL Ensures we can use SDL to see the game screen
# -D_GNU_SOURCE=1 means the compiler will use the GNU standard of compilation, the superset of all other standards under GNU C libraries.
# -D_REENTRANT causes the compiler to use thread safe (i.e. re-entrant) versions of several functions in the C library.
CFLAGS := -c -O3 -I$(ALE)/src -I/opt/local/include -D_GNU_SOURCE=1 -D_REENTRANT -std=c++11 -Wno-deprecated-declarations -Wall -g

EXE := learner
# Search for library 'ale' and library 'z' when linking.
LDFLAGS := -L$(ALE) -lale -lz -lm

ifeq ($(strip $(USE_SDL)), 1)
  FLAGS +=  -D__USE_SDL `sdl-config --cflags --libs`
  LDFLAGS += -lSDL
endif

#This variable contains the paths in which there are files to compile (separated by a : )
VPATH = common:features:agents/rl/sarsa:agents/rl/qlearning:agents/baseline:agents/human

SRCS = main_grid_example.cpp Mathematics.cpp Parameters.cpp Timer.cpp Features.cpp Background.cpp BasicFeatures.cpp BASSFeatures.cpp BPROFeatures.cpp RAMFeatures.cpp SarsaLearner.cpp QLearner.cpp ConstantAgent.cpp PerturbAgent.cpp  RandomAgent.cpp  HumanAgent.cpp #TRSarsaLearner.cpp
OBJS = $(subst .cpp,.o, $(addprefix bin/,$(SRCS)))

all: $(EXE)

$(EXE): $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o $(EXE)

bin/%.o: %.cpp
	$(CC) -o $@ -c $< $(CFLAGS)

clean:
	rm -rf $(OBJS)

#This command needs to be executed in a osX before running the code:
#export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:../lib/ale_0_4"
