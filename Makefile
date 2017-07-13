EXE = gppKerCpp
SRC = gppKer.cpp 

CXX = CC

CXXFLAGS = -O3 -qopt-report=5 -g -xCORE_AVX2 -qopenmp
LINK = ${CXX}
LINKFLAGS = -dynamic -O3 -qopenmp

OBJ = $(SRC:.cpp=.o)

$(EXE): $(OBJ) 
	$(CXX) $(OBJ) -o $(EXE) $(LINKFLAGS)

$(OBJ): $(SRC) 
	$(CXX) -c $(SRC) $(CXXFLAGS)

clean: 
	rm -f *.o gppKerCpp
