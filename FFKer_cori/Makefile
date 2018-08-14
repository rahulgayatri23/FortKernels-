EXE = ffCC_Cuda.ex
SRC1 = ffCustomComplex.cpp 
SRC2 = ffCC_Cuda.cu

CXX = nvcc

LINK = ${CXX}

ifeq ($(CXX),nvcc)
	CXXFLAGS= -g -O3 -std=c++11 -Wno-deprecated-gpu-targets 
#	CXXFLAGS+= -Xptxas -v --maxrregcount=150
 #   CXXFLAG+=-arch=sm_35
    CXXFLAGS+=-arch=sm_60
	LINKFLAGS=-Wno-deprecated-gpu-targets
endif 

OBJ1 = $(SRC1:.cpp=.o)
OBJ2 = $(SRC2:.cu=.o)

$(EXE): $(OBJ1) $(OBJ2) 
	$(CXX) $(OBJ1) $(OBJ2) -o $(EXE) $(LINKFLAGS)

$(OBJ1): $(SRC1) 
	$(CXX) -c $(SRC1) $(CXXFLAGS)

$(OBJ2): $(SRC2) 
	$(CXX) -c $(SRC2) $(CXXFLAGS)

clean: 
	rm -f $(OBJ1) $(OBJ2) $(EXE)

