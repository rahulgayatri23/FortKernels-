EXE = gppKerCpp
SRC = gppKer.cpp 

CXX = g++

CXXFLAGS = -O3 -fopenmp -std=c++11 #-qopt-report=5 -g -qopenmp -xCORE_AVX2 #  -xmic_avx512 #-no-vec
LINK = ${CXX}
LINKFLAGS = -O3 -fopenmp

OBJ = $(SRC:.cpp=.o)

$(EXE): $(OBJ) 
	$(CXX) $(OBJ) -o $(EXE) $(LINKFLAGS)

$(OBJ): $(SRC) 
	$(CXX) -c $(SRC) $(CXXFLAGS)

clean: 
	rm -f *.o gppKerCpp
