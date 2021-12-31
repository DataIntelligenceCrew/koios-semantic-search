# Koios : Top-k Table Join Search using Semantic Overlap
 
 Code for Koios  : Main-baseline.cpp

 Code for Koios+ : Main-clean.cpp 

## Dependencies

Cmake version 3.18 (version important):

	-If having older version installed:
		apt remove --purge --auto-remove cmake



Faiss index by Facebook:

	-see [INSTALL.md](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md) for details
	-For now, if encountering any CUDA error, please use -DFAISS_ENABLE_GPU=OFF when compiling faiss
	-Step 3: sudo make install "is not optional"

Sqlite3:

	apt-get install sqlite3 libsqlite3-dev

FastText:

	- Use thee following API to generate the FastTextDB https://github.com/ekzhu/go-fasttext

Warning: do not change Makefile flag orders [!!!]

Setting up:


	- run source bashrc
	- . /opt/intel/oneapi/setvars.sh --config=intel.config
	- set the variable **setloc** in the cpp file to the data lake location
	- make 
	- Note update Makefile to change from Koios to Koios+