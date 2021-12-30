CC = g++
LINK1 = $${HOME}/opt/usr/local/lib
LINK2 = /usr/local/cuda/lib64
HUNG = ./hungarian-algorithm-cpp-master
MOD = ./modules
USER_INC = $${HOME}/opt/usr/local/include
CUDA_INC = /usr/local/cuda/include
MOD = ./modules
INC_DIR = $(HUNG) $(USER_INC) $(MOD)
LINK_DIR = $(LINK1) $(LINK2)
INC_PARAMS = $(foreach d, $(INC_DIR), -I$d)
LINK_PARAMS = $(foreach l, $(LINK_DIR), -L$l)
CFLAGS = -lstdc++fs -fPIC -std=c++17 -Wno-deprecated -fopenmp $(LINK_PARAMS) $(INC_PARAMS) -lortools -lsqlite3 -lcuda -lcudart -lcublas -lfaiss -lopenblas -llapack -ltbb

main-baseline: main-baseline.o hung.o timing.o
	$(CC) -g -o main-baseline main-baseline.o hung.o timing.o $(CFLAGS)

main-baseline.o: Main-baseline.cpp
	$(CC) -g -c Main-baseline.cpp $(CFLAGS) -o main-baseline.o

hung.o: $(HUNG)/Hungarian.cpp $(HUNG)/Hungarian.h
	$(CC) -g -c $(HUNG)/Hungarian.cpp -o hung.o

timing.o: $(MOD)/timing.cxx $(MOD)/timing.h
	$(CC) -g -c $(MOD)/timing.cxx -o timing.o







