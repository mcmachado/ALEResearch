# Makefile
#
# Author: Marlos C. Machado, Nicolas Carion

CXX := g++

ALE := ../../Arcade-Learning-Environment

# Set this to 1 to enable SDL and display_screen
USE_SDL     := 0

# -O3 Optimize code (turns on all optimizations specified by -O2 and also turns on the -finline-functions, -funswitch-loops, -fpredictive-commoning, -fgcse-after-reload, -ftree-loop-vectorize, -ftree-slp-vectorize, -fvect-cost-model, -ftree-partial-pre and -fipa-cp-clone options).
# -D__USE_SDL Ensures we can use SDL to see the game screen
# -D_GNU_SOURCE=1 means the compiler will use the GNU standard of compilation, the superset of all other standards under GNU C libraries.
# -D_REENTRANT causes the compiler to use thread safe (i.e. re-entrant) versions of several functions in the C library.
CFLAGS := -O3 -I$(ALE)/src -I/opt/local/include -D_GNU_SOURCE=1 -D_REENTRANT -std=c++11 -Wno-deprecated-declarations -Wall -g -MMD

EXE := learner
GETTER := mode_difficulty_getter
VALUE_TRANSFER := value_transfer
VALUE_TRANSFER_RELEARN := value_transfer_relearn
OFFPOLICY_LEARN := offpolicy_learn
SVD := svd
# Search for library 'ale' and library 'z' when linking.
LDFLAGS := -L$(ALE) -lale -lz -lm

ifeq ($(strip $(USE_SDL)), 1)
  CFLAGS +=  -D__USE_SDL `sdl-config --cflags --libs`
  LDFLAGS += -lSDL
endif

#This variable contains the paths in which there are files to compile (separated by a : )
VPATH = common:features:agents/rl/sarsa:agents/rl/qlearning:agents/baseline:agents/human:offPolicy

SRCS = Mathematics.cpp Parameters.cpp Timer.cpp Features.cpp Background.cpp BasicFeatures.cpp BASSFeatures.cpp BPROFeatures.cpp RAMFeatures.cpp SarsaLearner.cpp QLearner.cpp ConstantAgent.cpp PerturbAgent.cpp  RandomAgent.cpp  HumanAgent.cpp GQLearner.cpp OffPolicyLearner.cpp #TRSarsaLearner.cpp
OBJS = $(subst .cpp,.o, $(addprefix bin/,$(SRCS)))
DEPS = $(subst .cpp,.d, $(addprefix bin/,$(SRCS))) bin/main_grid_example.d bin/main_offpolicy.d

all: $(EXE) $(GETTER) $(VALUE_TRANSFER) $(VALUE_TRANSFER_RELEARN) $(OFFPOLICY_LEARN) $(SVD)

$(EXE): $(OBJS) bin/main_sarsa_example.o
	$(CXX) $(LDFLAGS) $(OBJS) bin/main_sarsa_example.o -o $(EXE)

$(GETTER):
	$(CXX) $(LDFLAGS) $(CFLAGS) mode_difficulty_getter.cpp -o $(GETTER)

$(VALUE_TRANSFER): $(OBJS) bin/value_transfer.o
	$(CXX) $(LDFLAGS) $(CFLAGS) $(OBJS) bin/value_transfer.o -o $(VALUE_TRANSFER)

$(VALUE_TRANSFER_RELEARN): $(OBJS) bin/value_transfer_relearn.o
	$(CXX) $(LDFLAGS) $(CFLAGS) $(OBJS) bin/value_transfer_relearn.o -o $(VALUE_TRANSFER_RELEARN)

$(OFFPOLICY_LEARN): $(OBJS) bin/main_offpolicy.o
	$(CXX) $(LDFLAGS) $(CFLAGS) $(OBJS) bin/main_offpolicy.o -o $(OFFPOLICY_LEARN)

$(SVD): $(OBJS) bin/svd.o
	$(CXX) $(LDFLAGS) $(CFLAGS) $(OBJS) bin/svd.o -o $(SVD)


bin/%.o: %.cpp
	$(CXX) -o $@ -c $< $(CFLAGS)

clean:
	rm -rf $(OBJS) $(EXE) $(DEPS) $(VALUE_TRANSFER) $(GETTER) $(SVD) bin/*

-include $(OBJS:%.o=%.d) #for handling dependencies between files


#This command needs to be executed in a osX before running the code:
#export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:../lib/ale_0_4"
