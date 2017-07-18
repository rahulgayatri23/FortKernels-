EXE = gppKerCpp
SRC = gppKer.cpp 

CXX = CC

CXXFLAGS = -O3 -qopt-report=5 -g -qopenmp -xCORE_AVX2 #  -xmic_avx512 #-no-vec
LINK = ${CXX}
LINKFLAGS = -dynamic -O3 -qopenmp

OBJ = $(SRC:.cpp=.o)

$(EXE): $(OBJ) 
	$(CXX) $(OBJ) -o $(EXE) $(LINKFLAGS)

$(OBJ): $(SRC) 
	$(CXX) -c $(SRC) $(CXXFLAGS)

clean: 
	rm -f *.o gppKerCpp
