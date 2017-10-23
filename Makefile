EXE = gppKer_gpuComplexOpenMP3.ex
SRC = gppKer_gpuComplexOpenMP3.cpp 
SRC+=Complex.h

#CXX = g++
#CXX = xlc++
CXX = CC

LINK = ${CXX}

ifeq ($(CXX),g++)
	CXXFLAGS=-g -O3 -std=c++11 -fopenmp
	LINKFLAGS=-fopenmp
endif 

ifeq ($(CXX),xlc++)
	CXXFLAGS=-O3 -std=gnu++11 -g -qsmp
	LINKFLAGS=-qsmp
endif 

ifeq ($(CXX),CC)
	CXXFLAGS=-O3 -qopenmp -std=c++11 -qopt-report=5
	CXXFLAGS+=-xCORE_AVX2
#	CXXFLAGS+=-xMIC_AVX512
	LINKFLAGS=-qopenmp -std=c++11
endif 

OBJ = $(SRC:.cpp=.o)

$(EXE): $(OBJ) 
	$(CXX) $(OBJ) -o $(EXE) $(LINKFLAGS)

$(OBJ): $(SRC) 
	$(CXX) -c $(SRC) $(CXXFLAGS)

clean: 
	rm -f *.o $(EXE) 
