#include <string>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>

namespace fs = std::filesystem;

using namespace std;

/**************************
 * Environmental variables 
 **************************/
class Environment {
	private:
		unordered_map<string, unordered_set<string>> sets;
		unordered_set<string> wordset;
		unordered_map<string, unordered_set<string>> rindex;
		int average_cardinality = 0;

	public:
		Environment(string, int);
		unordered_map<string, unordered_set<string>> getSets();
		unordered_set<string> getWordset();
		unordered_map<string, unordered_set<string>> getRindex();
		int getAvg();
};