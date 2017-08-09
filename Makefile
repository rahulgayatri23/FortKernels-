EXE = gppKerCpp
SRC = gppKer.cpp 

CXX = icc

CXXFLAGS = -O3 -std=c++11
CXXFLAGS += -qopenmp 
CXXFLAGS += -xCORE_AVX2
#CXXFLAGS += -xMIC_AVX512

LINK = ${CXX}
LINKFLAGS = -O3 -qopenmp

OBJ = $(SRC:.cpp=.o)

$(EXE): $(OBJ) 
	$(CXX) $(OBJ) -o $(EXE) $(LINKFLAGS)

$(OBJ): $(SRC) 
	$(CXX) -c $(SRC) $(CXXFLAGS)

clean: 
	rm -f *.o gppKerCpp
