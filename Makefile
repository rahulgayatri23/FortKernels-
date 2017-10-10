EXE = gppKer.ex
SRC1 = gppKer.cpp 

CXX = xlc++
#CXX = CC 
#CXX = g++
#CXX = clang++

LINK = ${CXX}

ifeq ($(CXX),g++)
	CXXFLAGS= -g -O3 -std=c++11 -fopenmp -foffload="-lm" -foffload=nvptx-none
	LINKFLAGS=-fopenmp
endif 

ifeq ($(CXX),xlc++)
#	CXXFLAGS=-O3 -std=gnu++11 -g -qsmp
#	LINKFLAGS=-qsmp
	CXXFLAGS=-O3 -std=gnu++11 -g -qsmp=noauto:omp -qoffload #-Xptxas -v
	LINKFLAGS=-qsmp=noauto:omp -qoffload 
endif 

ifeq ($(CXX),clang++)
	CXXFLAGS=-O3 -std=gnu++11 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_HOME}
	LINKFLAGS=-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_HOME}
endif 

ifeq ($(CXX),icc)
	CXXFLAGS=-O3 -qopenmp -qopt-report=5
	CXXFLAGS+=xCORE_AVX2
#	CXXFLAGS+=-xMIC_AVX512
	LINKFLAGS=-qopenmp
endif 

OBJ1 = $(SRC1:.cpp=.o)

$(EXE): $(OBJ1)  
	$(CXX) $(OBJ1) $(OBJ2) -o $(EXE) $(LINKFLAGS)

$(OBJ1): $(SRC1) 
	$(CXX) -c $(SRC1) $(CXXFLAGS)

clean: 
	rm -f $(OBJ1) $(EXE)

