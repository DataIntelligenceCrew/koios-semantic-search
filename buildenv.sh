echo "Building environments"
source env.list
echo "added library paths"
. /opt/intel/oneapi/setvars.sh --config=intel.config
echo "initialized intel tbb"
