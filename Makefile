#EXE = gppKerOpenMPFor.ex
#SRC = gppKerOpenMPFor_xl.cpp 
#SRC = gppKerOpenMPFor.cpp 
SRC = gppKer_real.cpp 
EXE = gppKer_real.ex

#CXX = xlc++
CXX = CC 
##CXX = g++
##CXX = clang++
#
#LINK = ${CXX}
#
#ifeq ($(CXX),g++)
#	CXXFLAGS= -g -O3 -std=c++11 -fopenmp 
#	LINKFLAGS=-fopenmp
#endif 
#
#ifeq ($(CXX),xlc++)
#	CXXFLAGS=-O3 -std=gnu++11 -g -qsmp
#	LINKFLAGS=-qsmp
#endif 
#
#ifeq ($(CXX),clang++)
#	CXXFLAGS=-O3 -std=gnu++11 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_HOME}
#	LINKFLAGS=-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_HOME}
#endif 
#
#ifeq ($(CXX),CC)
	CXXFLAGS=-O3 -qopenmp -qopt-report=5 -std=c++11 -qopenmp
#	CXXFLAGS+=-xCORE_AVX2
	CXXFLAGS+=-xMIC_AVX512
	LINKFLAGS=-qopenmp
#endif 


OBJ1 = $(SRC:.cpp=.o)

$(EXE): $(OBJ1)  
	$(CXX) $(OBJ1) -o $(EXE) $(LINKFLAGS)

$(OBJ1): $(SRC) 
	$(CXX) -c $(SRC) $(CXXFLAGS)

clean: 
	rm -f $(OBJ1) $(EXE)

