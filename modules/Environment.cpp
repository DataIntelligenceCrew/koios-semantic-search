#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include "Environment.h"

namespace fs = std::filesystem;

using namespace std;

/**************************
 * Environmental variables 
 **************************/

Environment::Environment(string lake, int size) {
	cout << "creating enviroment with lake ";
	cout << lake << " at size ";
	cout << size << endl;
	vector<string> files;
	int tokens = 0;
	for (const auto &entry: fs::directory_iterator(lake)) {
		files.push_back(entry.path());
	}
	cout << files.size() << " files listed" << endl;
	for (size_t i = 0; i < files.size(); ++i) {
		string f = files[i];
		if (f.find("fra") != string::npos) {
			continue;
		}
		string line;
		ifstream infile(f);
		if (infile.is_open()) {
			string line;
			// string newline = "\n";
			while (getline(infile, line)) {
				tokens += 1;
				// //replace newline
				// size_t pos = line.find(newline);
				// if (pos != string::npos) {
				// 	line.erase(pos, newline.length()); 
				// }
				line.erase(line.find_last_not_of(" \n\r\t")+1);
				if (line.size() > 0){
					sets[f].insert(line);
				}
				if (rindex.find(line) == rindex.end()){
					//not found
					wordset.insert(line);
					rindex[line].insert(f);
				} else {
					//found
					rindex[line].insert(f);
				}
			}
		}
		if (sets.size() >= size) {
			break;
		}
	}
	average_cardinality = tokens / sets.size();
}

unordered_map<string, unordered_set<string>> Environment::getSets() {
	return sets;
}

unordered_set<string> Environment::getWordset() {
	return wordset;
}

unordered_map<string, unordered_set<string>> Environment::getRindex() {
	return rindex;
}

int Environment::getAvg() {
	return average_cardinality;
}
