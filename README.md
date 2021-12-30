# koios-semantic-search

## Dependencies

Cmake version 3.18 (version important):

	-If having older version installed:
		apt remove --purge --auto-remove cmake

	-Then:
		mkdir ~/temp
		cd ~/temp
		wget https://cmake.org/files/v3.18/cmake-3.18.2.tar.gz
		tar -xzvf cmake-3.18.2.tar.gz
		cd cmake-3.18.2/
		./bootstrap
		make -j$(nproc)
		sudo make install

Openmp:

	apt-get install libomp-dev

Blas and Lapack:

	apt-get install libblas-dev liblapack-dev

Swig:

	apt-get install -y swig

Faiss index by Facebook:

	-see [INSTALL.md](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md) for details
	-For now, if encountering any CUDA error, please use -DFAISS_ENABLE_GPU=OFF when compiling faiss
	-Step 3: sudo make install "is not optional"

Sqlite3:

	apt-get install sqlite3 libsqlite3-dev

Warning: do not change Makefile flag orders [!!!]