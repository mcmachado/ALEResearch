# Author: Marlos C. Machado

ALE := ../../../../MyALE/

# Set this to 1 to enable SDL and display_screen
USE_SDL     := 0

# -O3 Optimize code (urns on all optimizations specified by -O2 and also turns on the -finline-functions, -funswitch-loops, -fpredictive-commoning, -fgcse-after-reload, -ftree-loop-vectorize, -ftree-slp-vectorize, -fvect-cost-model, -ftree-partial-pre and -fipa-cp-clone options).
FLAGS := -O3 -I$(ALE)/src
CXX := g++ -std=c++11
OUT_FILE := pfd
# Search for library 'ale' and library 'z' when linking.
LDFLAGS := -lale -lz -lm

ifeq ($(strip $(USE_SDL)), 1)
  FLAGS +=  -D__USE_SDL `sdl-config --cflags --libs`
  LDFLAGS += -lSDL -lSDL_gfx -lSDL_image 
endif

all: pfd

pfd:                bin/main.o bin/Parameters.o bin/ControlAgent.o bin/DimReduction.o bin/Background.o bin/RAMFeatures.o bin/BPROFeatures.o
	$(CXX) $(FLAGS) -L$(ALE) bin/main.o bin/Parameters.o bin/ControlAgent.o bin/DimReduction.o bin/Background.o bin/RAMFeatures.o bin/BPROFeatures.o $(LDFLAGS) -o $(OUT_FILE) 

bin/main.o: main.cpp
	$(CXX) $(FLAGS) -c main.cpp -o bin/main.o

bin/Parameters.o:      input/Parameters.cpp
	$(CXX) $(FLAGS) -c input/Parameters.cpp -o bin/Parameters.o

bin/Background.o:      observations/Background.cpp
	$(CXX) $(FLAGS) -c observations/Background.cpp -o bin/Background.o

bin/RAMFeatures.o:     observations/RAMFeatures.cpp
	$(CXX) $(FLAGS) -c observations/RAMFeatures.cpp -o bin/RAMFeatures.o

bin/BPROFeatures.o:    observations/BPROFeatures.cpp
	$(CXX) $(FLAGS) -c observations/BPROFeatures.cpp -o bin/BPROFeatures.o

bin/ControlAgent.o:    control/ControlAgent.cpp
	$(CXX) $(FLAGS) -c control/ControlAgent.cpp -o bin/ControlAgent.o

bin/DimReduction.o:    svd/DimReduction.cpp
	$(CXX) $(FLAGS) -c svd/DimReduction.cpp -o bin/DimReduction.o

clean:
	rm -rf ${OUT_FILE} bin/*.o


#This command needs to be executed in a osX before running the code:
#export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:../../../../MyALE/"
