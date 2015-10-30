# Makefile
#
# Author: Marlos C. Machado, Nicolas Carion

CXX := g++

ALE := ../../MyALE

# Set this to 1 to enable SDL and display_screen
USE_SDL     := 1

# -O3 Optimize code (urns on all optimizations specified by -O2 and also turns on the -finline-functions, -funswitch-loops, -fpredictive-commoning, -fgcse-after-reload, -ftree-loop-vectorize, -ftree-slp-vectorize, -fvect-cost-model, -ftree-partial-pre and -fipa-cp-clone options).
# -D__USE_SDL Ensures we can use SDL to see the game screen
# -D_GNU_SOURCE=1 means the compiler will use the GNU standard of compilation, the superset of all other standards under GNU C libraries.
# -D_REENTRANT causes the compiler to use thread safe (i.e. re-entrant) versions of several functions in the C library.
FLAGS := -O3 -I$(ALE)/src -D_GNU_SOURCE=1 -D_REENTRANT -std=c++11 -Wno-deprecated-declarations -Wall -g -MMD

MC := learnerMountainCar
EXE := learnerALE
GRID := learnerGridWorld
FAST := learnerFastSarsa
TEST := testEnvStepByStep
EXPL := learnerExplorationGridWorld

# Search for library 'ale' and library 'z' when linking.
LDFLAGS := -L$(ALE) -lale -lz -lm

ifeq ($(strip $(USE_SDL)), 1)
  FLAGS +=  -D__USE_SDL `sdl-config --cflags`
  LDFLAGS += -lSDL -lSDL_gfx -lSDL_image `sdl-config --cflags --libs`
endif

all: $(EXE) $(GRID) $(MC) $(FAST) $(TEST) ${EXPL}

mc: $(MC)
ale: $(EXE)
grid: $(GRID)
fast: ${FAST}
test: ${TEST}
expl: ${EXPL}

learnerALE: bin/main.o bin/Mathematics.o bin/Parameters.o bin/Timer.o bin/Features.o bin/Background.o bin/BasicFeatures.o bin/BASSFeatures.o bin/BPROFeatures.o bin/RAMFeatures.o bin/SarsaLearner.o bin/QLearner.o bin/TRSarsaLearner.o bin/FastSarsaLearner.o bin/RandomAgent.o bin/ConstantAgent.o bin/PerturbAgent.o bin/HumanAgent.o
	$(CXX) $(FLAGS) bin/main.o bin/Mathematics.o bin/Timer.o bin/Parameters.o bin/Features.o bin/Background.o bin/BasicFeatures.o bin/BASSFeatures.o bin/BPROFeatures.o bin/RAMFeatures.o bin/SarsaLearner.o bin/QLearner.o bin/TRSarsaLearner.o bin/FastSarsaLearner.o bin/RandomAgent.o bin/ConstantAgent.o bin/PerturbAgent.o bin/HumanAgent.o $(LDFLAGS) -o $(EXE)

learnerGridWorld: bin/main_grid_example.o bin/Mathematics.o bin/Parameters.o bin/Timer.o bin/Features.o bin/Background.o bin/BasicFeatures.o bin/BASSFeatures.o bin/BPROFeatures.o bin/RAMFeatures.o bin/SarsaLearner.o bin/QLearner.o bin/TRSarsaLearner.o bin/FastSarsaLearner.o bin/RandomAgent.o bin/ConstantAgent.o bin/PerturbAgent.o bin/HumanAgent.o
	$(CXX) $(FLAGS) bin/main_grid_example.o bin/Mathematics.o bin/Timer.o bin/Parameters.o bin/Features.o bin/Background.o bin/BasicFeatures.o bin/BASSFeatures.o bin/BPROFeatures.o bin/RAMFeatures.o bin/SarsaLearner.o bin/QLearner.o bin/TRSarsaLearner.o bin/FastSarsaLearner.o bin/RandomAgent.o bin/ConstantAgent.o bin/PerturbAgent.o bin/HumanAgent.o $(LDFLAGS) -o $(GRID)

learnerMountainCar: bin/main_mountain_car_example.o bin/Mathematics.o bin/Parameters.o bin/Timer.o bin/Features.o bin/Background.o bin/BasicFeatures.o bin/BASSFeatures.o bin/BPROFeatures.o bin/RAMFeatures.o bin/SarsaLearner.o bin/QLearner.o bin/TRSarsaLearner.o bin/FastSarsaLearner.o bin/RandomAgent.o bin/ConstantAgent.o bin/PerturbAgent.o bin/HumanAgent.o
	$(CXX) $(FLAGS) bin/main_mountain_car_example.o bin/Mathematics.o bin/Timer.o bin/Parameters.o bin/Features.o bin/Background.o bin/BasicFeatures.o bin/BASSFeatures.o bin/BPROFeatures.o bin/RAMFeatures.o bin/SarsaLearner.o bin/QLearner.o bin/TRSarsaLearner.o bin/FastSarsaLearner.o bin/RandomAgent.o bin/ConstantAgent.o bin/PerturbAgent.o bin/HumanAgent.o $(LDFLAGS) -o $(MC)

learnerFastSarsa: bin/main_mc_fastSarsa.o bin/Mathematics.o bin/Parameters.o bin/Timer.o bin/Features.o bin/Background.o bin/BasicFeatures.o bin/BASSFeatures.o bin/BPROFeatures.o bin/RAMFeatures.o bin/SarsaLearner.o bin/QLearner.o bin/TRSarsaLearner.o bin/FastSarsaLearner.o bin/RandomAgent.o bin/ConstantAgent.o bin/PerturbAgent.o bin/HumanAgent.o
	$(CXX) $(FLAGS) bin/main_mc_fastSarsa.o bin/Mathematics.o bin/Timer.o bin/Parameters.o bin/Features.o bin/Background.o bin/BasicFeatures.o bin/BASSFeatures.o bin/BPROFeatures.o bin/RAMFeatures.o bin/SarsaLearner.o bin/QLearner.o bin/TRSarsaLearner.o bin/RandomAgent.o bin/FastSarsaLearner.o bin/ConstantAgent.o bin/PerturbAgent.o bin/HumanAgent.o $(LDFLAGS) -o $(FAST)

testEnvStepByStep: bin/testEnvironmentStepByStep.o bin/Parameters.o bin/Features.o bin/Background.o bin/BasicFeatures.o bin/BASSFeatures.o bin/BPROFeatures.o bin/RAMFeatures.o
	$(CXX) $(FLAGS) bin/testEnvironmentStepByStep.o bin/Mathematics.o bin/Timer.o bin/Parameters.o bin/Features.o bin/Background.o bin/BasicFeatures.o bin/BASSFeatures.o bin/BPROFeatures.o bin/RAMFeatures.o $(LDFLAGS) -o $(TEST)

learnerExplorationGridWorld: bin/main_grid_exploration.o bin/Mathematics.o bin/Parameters.o bin/Timer.o bin/Features.o bin/Background.o bin/BasicFeatures.o bin/BASSFeatures.o bin/BPROFeatures.o bin/RAMFeatures.o bin/SarsaLearner.o bin/QLearner.o bin/TRSarsaLearner.o bin/FastSarsaLearner.o bin/SarsaSplitLearner.o bin/RandomAgent.o bin/ConstantAgent.o bin/PerturbAgent.o bin/HumanAgent.o
	$(CXX) $(FLAGS) bin/main_grid_exploration.o bin/Mathematics.o bin/Timer.o bin/Parameters.o bin/Features.o bin/Background.o bin/BasicFeatures.o bin/BASSFeatures.o bin/BPROFeatures.o bin/RAMFeatures.o bin/SarsaLearner.o bin/QLearner.o bin/TRSarsaLearner.o bin/FastSarsaLearner.o bin/SarsaSplitLearner.o bin/RandomAgent.o bin/ConstantAgent.o bin/PerturbAgent.o bin/HumanAgent.o $(LDFLAGS) -o $(EXPL)

bin/main.o: main.cpp
	$(CXX) $(FLAGS) -c main.cpp -o bin/main.o

bin/main_grid_example.o: main_grid_example.cpp
	$(CXX) $(FLAGS) -c main_grid_example.cpp -o bin/main_grid_example.o

bin/main_mountain_car_example.o: main_mountain_car_example.cpp
	$(CXX) $(FLAGS) -c main_mountain_car_example.cpp -o bin/main_mountain_car_example.o

bin/main_mc_fastSarsa.o: main_mc_fastSarsa.cpp
	$(CXX) $(FLAGS) -c main_mc_fastSarsa.cpp -o bin/main_mc_fastSarsa.o

bin/main_grid_exploration.o: main_grid_exploration.cpp
	$(CXX) $(FLAGS) -c main_grid_exploration.cpp -o bin/main_grid_exploration.o

bin/testEnvironmentStepByStep.o: testEnvironmentStepByStep.cpp
	$(CXX) $(FLAGS) -c testEnvironmentStepByStep.cpp -o bin/testEnvironmentStepByStep.o

bin/Mathematics.o: common/Mathematics.cpp
	$(CXX) $(FLAGS) -c common/Mathematics.cpp -o bin/Mathematics.o

bin/Timer.o: common/Timer.cpp
	$(CXX) $(FLAGS) -c common/Timer.cpp -o bin/Timer.o

bin/Parameters.o: common/Parameters.cpp
	$(CXX) $(FLAGS) -c common/Parameters.cpp -o bin/Parameters.o

bin/Features.o: features/Features.cpp
	$(CXX) $(FLAGS) -c features/Features.cpp -o bin/Features.o

bin/Background.o: features/Background.cpp
	$(CXX) $(FLAGS) -c features/Background.cpp -o bin/Background.o
	
bin/BasicFeatures.o: features/BasicFeatures.cpp
	$(CXX) $(FLAGS) -c features/BasicFeatures.cpp -o bin/BasicFeatures.o

bin/BASSFeatures.o: features/BASSFeatures.cpp
	$(CXX) $(FLAGS) -c features/BASSFeatures.cpp -o bin/BASSFeatures.o

bin/BPROFeatures.o: features/BPROFeatures.cpp
	$(CXX) $(FLAGS) -c features/BPROFeatures.cpp -o bin/BPROFeatures.o	

bin/RAMFeatures.o: features/RAMFeatures.cpp
	$(CXX) $(FLAGS) -c features/RAMFeatures.cpp -o bin/RAMFeatures.o	

bin/SarsaLearner.o: agents/rl/sarsa/SarsaLearner.cpp
	$(CXX) $(FLAGS) -c agents/rl/sarsa/SarsaLearner.cpp -o bin/SarsaLearner.o

bin/QLearner.o: agents/rl/qlearning/QLearner.cpp
	$(CXX) $(FLAGS) -c agents/rl/qlearning/QLearner.cpp -o bin/QLearner.o

bin/TRSarsaLearner.o: agents/rl/true_online_sarsa/TrueOnlineSarsaLearner.cpp
	$(CXX) $(FLAGS) -c agents/rl/true_online_sarsa/TrueOnlineSarsaLearner.cpp -o bin/TRSarsaLearner.o

bin/FastSarsaLearner.o: agents/rl/fastSarsa/FastSarsaLearner.cpp
	$(CXX) $(FLAGS) -c agents/rl/fastSarsa/FastSarsaLearner.cpp -o bin/FastSarsaLearner.o

bin/SarsaSplitLearner.o: agents/rl/sarsa_split/SarsaSplitLearner.cpp
	$(CXX) $(FLAGS) -c agents/rl/sarsa_split/SarsaSplitLearner.cpp -o bin/SarsaSplitLearner.o

bin/RandomAgent.o: agents/baseline/RandomAgent.cpp
	$(CXX) $(FLAGS) -c agents/baseline/RandomAgent.cpp -o bin/RandomAgent.o	

bin/ConstantAgent.o: agents/baseline/ConstantAgent.cpp
	$(CXX) $(FLAGS) -c agents/baseline/ConstantAgent.cpp -o bin/ConstantAgent.o	

bin/PerturbAgent.o: agents/baseline/PerturbAgent.cpp
	$(CXX) $(FLAGS) -c agents/baseline/PerturbAgent.cpp -o bin/PerturbAgent.o		

bin/HumanAgent.o: agents/human/HumanAgent.cpp
	$(CXX) $(FLAGS) -c agents/human/HumanAgent.cpp -o bin/HumanAgent.o

clean:
	rm -rf $(OBJS) $(EXE) bin/*

#This command needs to be executed in a osX before running the code:
#export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:../lib/ale_0_4"
