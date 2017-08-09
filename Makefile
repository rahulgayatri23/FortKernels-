EXE=gppKerCpp
SRC=gppKer.cpp

#CXX=/Users/rgayatri/Documents/Softs/GCC/gcc-7.1.0/_build/bin/g++
#CXX=CC
CXX=icc

CXXFLAGS=-O3 -std=c++11 -qopt-report=5 -g -qopenmp 
CXXFLAGS+= -xCORE_AVX2
#CXXFLAGS+= -xMIC_AVX512
LINK=${CXX}
LINKFLAGS=-O3 -qopenmp

OBJ=$(SRC:.cpp=.o)

$(EXE): $(OBJ)
	$(CXX) $(OBJ) -o $(EXE) $(LINKFLAGS)

$(OBJ): $(SRC)
	$(CXX) -c $(SRC) $(CXXFLAGS)

clean:
	rm -f *.o gppKerCpp
