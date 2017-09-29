EXE = gppKer_gpuComplexOpenMP3.ex
SRC = gppKer_gpuComplexOpenMP3.cpp 
SRC+=Complex.h

CXX = g++
#CXX = xlc++
#CXX = icc

LINK = ${CXX}

ifeq ($(CXX),g++)
	CXXFLAGS=-g -O3 -std=c++11 -fopenmp
	LINKFLAGS=-fopenmp
endif 

ifeq ($(CXX),xlc++)
	CXXFLAGS=-O3 -std=gnu++11 -g -qsmp
	LINKFLAGS=-qsmp
endif 

ifeq ($(CXX),icc)
	CXXFLAGS=-O3 -qopenmp -qopt-report=5
	CXXFLAGS+=xCORE_AVX2
#	CXXFLAGS+=-xMIC_AVX512
	LINKFLAGS=-qopenmp
endif 

OBJ = $(SRC:.cpp=.o)

$(EXE): $(OBJ) 
	$(CXX) $(OBJ) -o $(EXE) $(LINKFLAGS)

$(OBJ): $(SRC) 
	$(CXX) -c $(SRC) $(CXXFLAGS)

clean: 
	rm -f *.o $(EXE) 
