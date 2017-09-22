EXE = gppKer_gpuComplex.ex
SRC = gppKer_gpuComplex.cpp 
SRC+=Complex.cpp

CXX = xlc++
#CXX = g++

LINK = ${CXX}


CXXFLAGS=-O3 -g -std=c++11 -qsmp -qoffload -g -Xptxas -v 
LINKFLAGS=-qsmp -qoffload
#CXXFLAGS= -g -foffload="-lm" -foffload=nvptx-none -O3 -std=c++11 -fopenmp#nvptx-none
#LINKFLAGS=-fopenmp

OBJ = $(SRC:.cpp=.o)

$(EXE): $(OBJ) 
	$(CXX) $(OBJ) -o $(EXE) $(LINKFLAGS)

$(OBJ): $(SRC) 
	$(CXX) -c $(SRC) $(CXXFLAGS)

clean: 
	rm -f *.o gppKer_gpuComplex.ex 
