# Koios : Top-k Table Join Search using Semantic Overlap
 
 Code for Koios  : Main-baseline.cpp

 Code for Koios+ : Main-clean.cpp 

## Dependencies

Cmake version 3.18 (version important):

	-If having older version installed:
		apt remove --purge --auto-remove cmake

Setting up:


	- run source bashrc
	- . /opt/intel/oneapi/setvars.sh --config=intel.config
	- make 
	- Note update Makefile to change from Koios to Koios+

Faiss index by Facebook:

	-see [INSTALL.md](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md) for details
	-For now, if encountering any CUDA error, please use -DFAISS_ENABLE_GPU=OFF when compiling faiss
	-Step 3: sudo make install "is not optional"

Sqlite3:

	apt-get install sqlite3 libsqlite3-dev

Warning: do not change Makefile flag orders [!!!]