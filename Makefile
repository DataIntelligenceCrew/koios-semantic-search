CC = g++
BUILD = ./build
SRC = ./src
OPT_LINK = ./opt/usr/local/lib
CUDA_LINK = /usr/local/cuda/lib64
HUNG = $(SRC)/hungarian-algorithm-cpp-master
MOD = ./modules
USER_INC = ./opt/usr/local/include
CUDA_INC = /usr/local/cuda/include
INC_DIR = $(HUNG) $(USER_INC) $(MOD)
LINK_DIR = $(OPT_LINK) $(CUDA_LINK)
INC_PARAMS = $(foreach d, $(INC_DIR), -I$d)
LINK_PARAMS = $(foreach l, $(LINK_DIR), -L$l)
CFLAGS = -lstdc++fs -fPIC -std=c++17 -Wno-deprecated -fopenmp $(LINK_PARAMS) $(INC_PARAMS) -lortools -lsqlite3 -lcuda -lcudart -lcublas -lfaiss -lopenblas -llapack -ltbb

koios-semantic: $(BUILD)/koios.o $(BUILD)/hung.o $(BUILD)/timing.o
	$(CC) -g -o $(BUILD)/koios-semantic $(BUILD)/koios.o $(BUILD)/hung.o $(BUILD)/timing.o $(CFLAGS)

baseline-semantic: $(BUILD)/baseline.o $(BUILD)/hung.o $(BUILD)/timing.o
	$(CC) -g -o $(BUILD)/baseline-semantic $(BUILD)/baseline.o $(BUILD)/hung.o $(BUILD)/timing.o $(CFLAGS)

$(BUILD)/baseline.o : $(SRC)/baseline.cpp
	$(CC) -g -c $ $(SRC)/baseline.cpp $(CFLAGS) -o $(BUILD)/baseline.o

$(BUILD)/koios.o : $(SRC)/koios.cpp
	$(CC) -g -c $ $(SRC)/koios.cpp $(CFLAGS) -o $(BUILD)/koios.o

$(BUILD)/hung.o: $(HUNG)/Hungarian.cpp $(HUNG)/Hungarian.h
	$(CC) -g -c $(HUNG)/Hungarian.cpp -o $(BUILD)/hung.o

$(BUILD)/timing.o: $(MOD)/timing.cxx $(MOD)/timing.h
	$(CC) -g -c $(MOD)/timing.cxx -o $(BUILD)/timing.o