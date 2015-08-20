# Makefile
#
# Author: Marlos C. Machado

ALE := ../../../../../MyALE

# Set this to 1 to enable SDL and display_screen
USE_SDL     := 0

# -O3 Optimize code
# -D_GNU_SOURCE=1 means the compiler will use the GNU standard of compilation (superset of all other standards under GNU C libraries).
# -D_REENTRANT causes the compiler to use thread safe versions of several functions in the C library.
FLAGS := -O3 -I$(ALE)/src -I/opt/local/include -D_GNU_SOURCE=1 -D_REENTRANT -D__STDC_CONSTANT_MACROS
CXX := g++ -std=c++0x
OUT_FILE := playOption
# Search for library 'ale' and library 'z' when linking.
LDFLAGS := -L$(ALE) -lale -lz -lm

ifeq ($(strip $(USE_SDL)), 1)
  FLAGS +=  -D__USE_SDL `sdl-config --cflags --libs`
  LDFLAGS += -lSDL -lSDL_gfx -lSDL_image 
endif

all: playOption

playOption:         bin/playOption.o bin/Parameters.o bin/control.o bin/Mathematics.o bin/Background.o bin/BPROFeatures.o
	$(CXX) $(FLAGS) bin/playOption.o bin/Parameters.o bin/control.o bin/Mathematics.o bin/Background.o bin/BPROFeatures.o $(LDFLAGS) -o $(OUT_FILE) 

bin/playOption.o: playOption.cpp
	$(CXX) $(FLAGS) -c playOption.cpp -o bin/playOption.o

bin/Parameters.o: Parameters.cpp
	$(CXX) $(FLAGS) -c Parameters.cpp -o bin/Parameters.o

bin/control.o: control.cpp
	$(CXX) $(FLAGS) -c control.cpp -o bin/control.o

bin/Mathematics.o: ../common/Mathematics.cpp
	$(CXX) $(FLAGS) -c ../common/Mathematics.cpp -o bin/Mathematics.o

bin/Background.o: ../features/Background.cpp
	$(CXX) $(FLAGS) -c ../features/Background.cpp -o bin/Background.o	

bin/BPROFeatures.o: ../features/BPROFeatures.cpp
	$(CXX) $(FLAGS) -c ../features/BPROFeatures.cpp -o bin/BPROFeatures.o

clean:
	rm -rf ${OUT_FILE} bin/*.o	

#This command needs to be executed in a osX before running the code:
#export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:../../../../MyALE/"
