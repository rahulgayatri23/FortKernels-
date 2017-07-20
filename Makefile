EXE = gppKerCpp
SRC = gppKer.cpp 

CXX = CC
CXXFLAGS = -O3 -g -fopenmp #--qopt-report=5 xCORE_AVX2 

LINK = ${CXX}
LINKFLAGS = -dynamic -O3 -fopenmp

OBJ = $(SRC:.cpp=.o)

$(EXE): $(OBJ) 
	$(CXX) $(OBJ) -o $(EXE) $(LINKFLAGS)

$(OBJ): $(SRC) 
	$(CXX) -c $(SRC) $(CXXFLAGS)

clean: 
	rm -f *.o gppKerCpp
