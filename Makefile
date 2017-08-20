EXE = gppKerCpp
SRC = gppKer.cpp 

#CXX = icc
#CXX = g++
#CXX = pgc++
CXX = CC

LINK = ${CXX}


##IF CC = intel
CXXFLAGS=-O3 -g -std=c++11 -qopenmp -qopt-report=5 -qopenmp-offload=host
#CXXFLAGS+=-xCORE_AVX2
CXXFLAGS+=-xMIC_AVX512
LINKFLAGS=-qopenmp -dynamic

##IF CC = gcc
#CXXFLAGS = -O3 -std=c++11 -fopenmp
#LINKFLAGS = -fopenmp

##else 
#    ifeq($(CXX), icc)
#    CXXFLAGS = -O3 -qopenmp -qopt-report=5
#    CXXFLAGS += xCORE_AVX2
##    CXXFLAGS += -xMIC_AVX512
#    LINKFLAGS = -qopenmp
##else
#    ifeq($(CXX), pgc++)
#    CXXFLAGS = -O3 -openmp    
#    LINKFLAGS = -openmp
#endif

OBJ = $(SRC:.cpp=.o)

$(EXE): $(OBJ) 
	$(CXX) $(OBJ) -o $(EXE) $(LINKFLAGS)

$(OBJ): $(SRC) 
	$(CXX) -c $(SRC) $(CXXFLAGS)

clean: 
	rm -f *.o gppKerCpp
