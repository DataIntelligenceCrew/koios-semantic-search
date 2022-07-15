# KOIOS

KOIOS is an efficient and exact filter verification framework to find the top-k sets with the maximum biparitie matching to a query set. Here we use KOIOS for semantic overlap search, where semantic overlap is the maximum biparite matching score between the tokens of the query set and the candidate set.

## Installation
- Clone the repository onto your local machine.
- Download the fasttext-database from [here](https://rochester.box.com/s/7nsiz3eo3y6lbsx7cp22seafve9evvu3), and save it in the root folder.
- Make sure all paths are correct in the `Makefile`
- Run the following commands to initialize environment and [Intel-OneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#gs.5sjj50):
```bash
source bashrc
. /opt/intel/oneapi/setvars.sh --config=intel.config
```

## Usage
> For Syntactic Overlap Search
```bash
make koios-semantic
./build/koios-semantic <data-lake-path> <query> <result-location> <sim-threshold> <k> <number-of-partitions> 1
```
> For Semantic Overlap Search using KOIOS
```bash
make koios-semantic
./build/koios-semantic <data-lake-path> <query> <result-location> <sim-threshold> <k> <number-of-partitions> 0
```
> For Semantic Overlap Search using Baseline
```bash
make baseline-semantic
./build/baseline-semantic <data-lake-path> <query> <result-location> <sim-threshold> <k> 
```

## Dependencies
>Cmake version 3.18 (version important):

If older version installed:
    `apt remove --purge --auto-remove cmake`

>Faiss index by Facebook:
- Refer [INSTALL.md](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md) for details
- For now, if encountering any CUDA error, please use -DFAISS_ENABLE_GPU=OFF when compiling faiss
- Step 3: sudo make install "is not optional"

>Sqlite3:

`apt-get install sqlite3 libsqlite3-dev`

FastText:

	- Use the following API to generate the FastTextDB https://github.com/ekzhu/go-fasttext

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)