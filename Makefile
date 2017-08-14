EXE = gppKerCpp
SRC = gppKer.cpp 

CXX = CC

CXXFLAGS = -O3 -std=c++11
CXXFLAGS += -qopenmp 
CXXFLAGS += -xCORE_AVX2 -qopt-report=5
#CXXFLAGS += -xMIC_AVX512 -qopt-report=5

LINK = ${CXX}
LINKFLAGS = -O3 -qopenmp

OBJ = $(SRC:.cpp=.o)

$(EXE): $(OBJ) 
	$(CXX) $(OBJ) -o $(EXE) $(LINKFLAGS)

$(OBJ): $(SRC) 
	$(CXX) -c $(SRC) $(CXXFLAGS)

clean: 
	rm -f *.o gppKerCpp
