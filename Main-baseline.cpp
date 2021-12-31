#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <filesystem>
#include <fstream>
#include <future>
#include <sqlite3.h>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <sstream>
#include <numeric>
#include <queue>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "hungarian-algorithm-cpp-master/Hungarian.h"
#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/utils/distances.h>
#include <faiss/index_io.h>
#include <thread>
#include <future>
#include <omp.h>
#include <regex>
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "ortools/algorithms/hungarian.h"
#include "absl/container/flat_hash_map.h"
#include "modules/timing.h"
#include <sys/wait.h>
#include <oneapi/tbb.h>
#include "thread_pool.hpp"

/*************************************** 
	SEMANTIC OVERLAP IMPLEMENTATION C++
****************************************/

std::mutex gmtx; // global mutex lock
namespace fs = std::filesystem;
using namespace std;
using idx_t = faiss::Index::idx_t;
typedef pair<double, int> ltuple; // double int pair template for top-K list
typedef pair<double, pair<int, int>> ptuple; // double <int, int> pair for ordered edge priority queue
faiss::gpu::StandardGpuResources res; // GPU resource global
inline size_t key(int i,int j) {return (size_t) i << 32 | (unsigned int) j;} // concat unsigned int with two integer set id as an edge's integer id

// decreasing comparator
struct cmp_decreasing {
	bool operator() (pair<double, int> lhs, pair<double, int> rhs) const {
		return std::get<0>(lhs) >= std::get<0>(rhs);
	}
};

/**************************
 * Environmental Variables 
 **************************/
class Environment {
	private:
		unordered_map<int, set<int>> sets; // sets located by id
		unordered_set<int> wordset; // full dictionary of tokens
		vector<set<int>> rindex; // inverted index
		// Two kind of integer ids
		// token id / set id
		unordered_map<int, string> int2word; // convert integer id to token
		unordered_map<string, int> word2int; // convert token back to integer id
		unordered_map<int, string> int2set; // convert integer id to set name
		unordered_map<string, int> set2int; // convert set name back to integer id
		int average_cardinality = 0;

	public:
		Environment(string lake, int size) {
			// Create new environment from data lake
			// Params:
			// lake: data lake files location
			// size: maximum size of data lake (if lake is larger than this value, trim happens)
			cout << "creating enviroment with lake ";
			cout << lake << " at size ";
			cout << size << endl;
			vector<string> files;
			int id = 0;
			int tokens = 0;
			int sid = 0;
			for (const auto &entry: fs::directory_iterator(lake)) {
				files.push_back(entry.path());
			}
			cout << files.size() << " files listed" << endl;
			for (size_t i = 0; i < files.size(); ++i) {
				string f = files[i];
				set2int[f] = sid;
				int2set[sid] = f;

				if (f.find("fra") != string::npos) {
					continue;
				}
				string line;
				ifstream infile(f);
				if (infile.is_open()) {
					string line;
					while (getline(infile, line)) {
						tokens += 1;
						line.erase(line.find_last_not_of(" \n\r\t")+1);
						if (line.size() > 0){
							if (word2int.find(line) == word2int.end()){
								word2int[line] = id;
								wordset.insert(id);
								int2word[id] = line;
								sets[sid].insert(id);
								std::set<int> nset = {sid};
								rindex.push_back(nset);
								id += 1;
							} else {
								sets[sid].insert(word2int[line]);
								rindex[word2int[line]].insert(sid);
							}
						}
					}
				}
				if (sets.size() >= size) {
					break;
				}
				sid += 1;
			}
			average_cardinality = tokens / sets.size();
		}
		unordered_map<int, set<int>> getSets() {
			return sets;
		}
		unordered_set<int> getWordset() {
			return wordset;
		}
		vector<set<int>> getRindex() {
			return rindex;
		}
		int getAvg() {
			return average_cardinality;
		}
		int toInt(string s) {
			if (word2int.find(s) == word2int.end()) {
				return -1;
			} else {
				return word2int[s];
			}
		}
		string toWord(int i) {
			return int2word[i];
		}
		int getSetId(string s) {
			if (set2int.find(s) == set2int.end()) {
				return -1;
			} else {
				return set2int[s];
			}
		}
		string getSetName(int i) {
			return int2set[i];
		} 
};

/********************************
 * Sqlite3 Instance and utilities 
 ********************************/
class Database {
	private:
		sqlite3 *db; // fastText database instance
		sqlite3 *dest; // database location
		vector<int> dictionary; // array of unique tokens, index is its integer id
		unordered_map<size_t, double> cache;
		Environment *env;

		// convert mysql entry from bytes to word vector
		vector<float> bytes_to_wv(const unsigned char *data, size_t dimension) {
			vector<float> result;
			for (size_t i = 0; i < dimension * 4; i = i+4) {
				float f;
				unsigned char b[] = {data[i], data[i+1], data[i+2], data[i+3]};
				memcpy(&f, &b, sizeof(f));
				// cout << f << endl;
				result.push_back(f);
			}
			return result;
		}

		// tool for investigating bytes
		void print_bytes(ostream& out, const unsigned char *data, size_t dataLen, bool format = true) {
			out << setfill('0');
			for(size_t i = 0; i < dataLen; ++i) {
				out << hex << setw(2) << (int)data[i];
				if (format) {
					out << (((i + 1) % 16 == 0) ? "\n" : " ");
				}
			}
			out << endl;
		}

		// tool for investigating vectors
		void print_vector(vector<float> v) {
			for (int i = 0; i < v.size(); ++i){
				cout << v[i] << ", ";
			}
			cout << "\n" << endl;
		}

		// cache for cosine similarity, not in use *
		bool cache_lookup(int qtoken, int ttoken) {
			return cache.find(key(qtoken, ttoken)) != cache.end();
		}

	public:
	// Database constructor
		Database(string path, Environment *e) {
			env = e;

			int rc = sqlite3_open(path.c_str(), &dest);
			if (rc) {
				cout << "Cannot open database:  " << sqlite3_errmsg(dest) << endl; 
				exit(0);
			} else {
				cout << "Successfully opened sqlite3 database" << endl;
			}

			// sqlite3 in-memory for performance
			rc = sqlite3_open(":memory:", &db);
			if (rc) {
				cout << "Cannot open in-memory database:  " << sqlite3_errmsg(db) << endl; 
				exit(0);
			} else {
				cout << "Successfully opened in-memory database" << endl;
			}

			sqlite3_backup *pBackup;
			pBackup = sqlite3_backup_init(db, "main", dest, "main");
			if( pBackup ){
				sqlite3_backup_step(pBackup, -1);
				sqlite3_backup_finish(pBackup);
			}
			rc = sqlite3_errcode(db);
			if (rc) {
				cout << "Cannot copy database:  " << sqlite3_errmsg(db) << endl; 
				exit(0);
			} else {
				cout << "Successfully copied to memory" << endl;
			}
			sqlite3_close(dest);
		}

		void terminate() {
			sqlite3_close(db);
			cout << "Successfully terminated sqlite3 database" << endl;
		}

		void clear_cache() {
			cache.clear();
		}

		bool isnum(string s) {
			size_t pos;
			string t = ".";
			while ((pos = s.find(t)) != string::npos) {
				s.erase(pos, t.length());
			}
			for (string::iterator it = s.begin(); it != s.end(); ++it) {
				if (!isdigit(*it)) {
					return false;
				}
			}
			return true;
		}

		// cosine similarity helper function
		double cosine_similarity(vector<float> A, vector<float> B, int vlength){
			float dot = 0.0, denom_a = 0.0, denom_b = 0.0 ;
			 for(int i = 0; i < vlength; ++i) {
				dot += A[i] * B[i] ;
				denom_a += A[i] * A[i] ;
				denom_b += B[i] * B[i] ;
			}
			return double(dot / (sqrt(denom_a) * sqrt(denom_b))) ;
		}

		// calculate cosine similarity between a pair of words
		float calculate_similarity(int src, int tgt) {
			int rc;

			string srcstring = env->toWord(src);
			string tgtstring = env->toWord(tgt);

			if (src == tgt) {
				return 1.0;
			}
			if (isnum(srcstring) || isnum(tgtstring)) {
				return 0.0;
			}
			if (cache_lookup(src, tgt)) {
				return cache[key(src, tgt)];
			}

			string querybase = "SELECT vec FROM wv WHERE word=?;";
			

			vector<float> vector1;
			vector<float> vector2;

			sqlite3_stmt *stmt = NULL;

			bool s = false;
			bool t = false;

			rc = sqlite3_prepare_v2(db, querybase.c_str(), -1, &stmt, NULL);
			rc = sqlite3_bind_text(stmt, 1, srcstring.c_str(), -1, SQLITE_TRANSIENT);
			if (rc != SQLITE_OK) {
				cout << "SELECT failed: " << sqlite3_errmsg(db) << endl;
			}
			while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
				unsigned char *bytes = (unsigned char*)sqlite3_column_blob(stmt, 0);
				vector1 = bytes_to_wv(bytes, 300);
				if (vector1.size() == 300) {
					s = true;
				}
			}
			sqlite3_finalize(stmt);

			rc = sqlite3_prepare_v2(db, querybase.c_str(), -1, &stmt, NULL);
			rc = sqlite3_bind_text(stmt, 1, tgtstring.c_str(), -1, SQLITE_TRANSIENT);
			if (rc != SQLITE_OK) {
				cout << "SELECT failed: " << sqlite3_errmsg(db) << endl;
			}
			while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
				unsigned char *bytes = (unsigned char*)sqlite3_column_blob(stmt, 0);
				vector2 = bytes_to_wv(bytes, 300);
				if (vector2.size() == 300) {
					t = true;
				}
			}
			sqlite3_finalize(stmt);

			if (!(s && t)) {
				return 0.0;
			} 
			double sim = cosine_similarity(vector1, vector2, 300);
			cache[key(src, tgt)] = sim;
			cache[key(tgt, src)] = sim;
			return sim;
		}

		// get normalized vector with token's integer id
		vector<float> get_normalized_vector(int tokenid){
			vector<float> result;
			string token = env->toWord(tokenid);
			int rc;
			stringstream ss;
			ss << "SELECT vec FROM wv WHERE word=\"" << token << "\";";
			string query = ss.str();
			string querybase = "SELECT vec FROM wv WHERE word=?;";
			sqlite3_stmt *stmt = NULL;
			rc = sqlite3_prepare(db, querybase.c_str(), -1, &stmt, NULL);
			rc = sqlite3_bind_text(stmt, 1, token.c_str(), -1, SQLITE_TRANSIENT);
			if (rc != SQLITE_OK) {
				cout << query << endl;
				cout << "SELECT failed: " << sqlite3_errmsg(db) << endl;
			} else {
				while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
					unsigned char *bytes = (unsigned char*)sqlite3_column_blob(stmt, 0);
					result = bytes_to_wv(bytes, 300);
				}
			}
			
			float* r = new float[300];
			r = result.data();
			faiss::fvec_renorm_L2(300, 1, r);
			vector<float> vr(r, r + 300);
			return vr;
		}

		// get vectors of words in valid set
		vector<vector<float>> get_valid_vectors(unordered_set<int> validset){
			vector<vector<float>> result;
			int rc;
			stringstream ss;
			ss << "SELECT word, vec FROM wv;";
			string query = ss.str();
			sqlite3_stmt *stmt = NULL;
			rc = sqlite3_prepare(db, query.c_str(), -1, &stmt, NULL);
			if (rc != SQLITE_OK) {
				cout << "SELECT failed: " << sqlite3_errmsg(db) << endl;
			}
			dictionary.clear();
			
			while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
				const char *word = (char*)sqlite3_column_text(stmt, 0);
				unsigned char *bytes = (unsigned char*)sqlite3_column_blob(stmt, 1);
				string str_t(word);
				// cout << str_t << endl;
				int wid = env->toInt(str_t);
				// cout << wid << endl;
				bool is_in = validset.find(wid) != validset.end();
				if (is_in) {
					dictionary.push_back(wid);
					vector<float> vector_t = bytes_to_wv(bytes, 300);
					result.push_back(vector_t);
				}
			}
			return result;
		}

		vector<int> get_dictionary(){
			return dictionary;
		}
		void clear_dictionary(){
			dictionary.clear();
		}
};

/******************************************
 * Matrix Creation with Specified Edges 
 ******************************************/
class ValidMatrix {
	private:
		vector<vector<double>> M; // the matrix
		int matching;
		int realsize;
	public:
		// building a valid matrix with query set and target set and valid edges
		// valid edges stand for edges popped from the monotonically decreasing priority queue by edge weights (cosine similarity)
		// by leveraging valid edges, 
		// we save computing resources by do not have to re-calculate cosine similarity for edges retrieved from the Faiss index (which is expensive)
		ValidMatrix(set<int> query_set, set<int> target_set, unordered_map<size_t, double> validedge) {
			int i = 0;
			set<int> query_set_pruned;
			set<int> target_set_pruned;
			matching = min(query_set.size(), target_set.size());
			for(set<int>::iterator itq = query_set.begin(); itq != query_set.end(); ++itq) {
				vector<double> temp;
				int qword = *itq;
				for(set<int>::iterator itt = target_set.begin(); itt != target_set.end(); ++itt) {
					int tword = *itt;
					if (validedge.find(key(qword, tword)) != validedge.end()) {
						query_set_pruned.insert(qword);
						target_set_pruned.insert(tword);
					} else {
						if (qword == tword) {
							query_set_pruned.insert(qword);
							target_set_pruned.insert(tword);
						}
					}// bucket upperbound not considering this
				}
			}
			realsize = query_set_pruned.size() * target_set_pruned.size();
			for(set<int>::iterator itq = query_set_pruned.begin(); itq != query_set_pruned.end(); ++itq) {
				vector<double> temp;
				int qword = *itq;
				for(set<int>::iterator itt = target_set_pruned.begin(); itt != target_set_pruned.end(); ++itt) {
					int tword = *itt;
					if (validedge.find(key(qword, tword)) != validedge.end()) {
						temp.push_back(0.0 - validedge[key(qword, tword)]);
					} else {
						if (qword == tword) {
							temp.push_back(-1.0);
						} else {
							temp.push_back(0.0);
						}
					}
				}
				M.push_back(temp);
			}
		}

		// toDo : do we need this?
		// polymorphed valid matrix builder with specified number of known exact matches (param: exacts)
		// intended to optimize, but not very helpful in practice
		ValidMatrix(set<int> query_set, set<int> target_set, unordered_map<size_t, double> validedge, set<int> exacts) {
			int i = 0;
			set<int> query_set_pruned;
			set<int> target_set_pruned;
			matching = min(query_set.size(), target_set.size());
			for(set<int>::iterator itq = query_set.begin(); itq != query_set.end(); ++itq) {
				vector<double> temp;
				int qword = *itq;
				for(set<int>::iterator itt = target_set.begin(); itt != target_set.end(); ++itt) {
					int tword = *itt;
					if (validedge.find(key(qword, tword)) != validedge.end()) {
						query_set_pruned.insert(qword);
						target_set_pruned.insert(tword);
					} else {
						if (qword == tword && exacts.find(qword) == exacts.end()) {
							query_set_pruned.insert(qword);
							target_set_pruned.insert(tword);
						}
					}// bucket upperbound not considering this
				}
			}
			realsize = query_set_pruned.size() * target_set_pruned.size();
			for(set<int>::iterator itq = query_set_pruned.begin(); itq != query_set_pruned.end(); ++itq) {
				vector<double> temp;
				int qword = *itq;
				for(set<int>::iterator itt = target_set_pruned.begin(); itt != target_set_pruned.end(); ++itt) {
					int tword = *itt;
					// cout << qword << endl;
					// cout << tword << endl;
					if (validedge.find(key(qword, tword)) != validedge.end()) {
						if (exacts.find(qword) == exacts.end() && exacts.find(tword) == exacts.end()){
							temp.push_back(0.0 - validedge[key(qword, tword)]);
						} else {
							temp.push_back(0.0);
						}
					} else {
						if (qword == tword && exacts.find(qword) == exacts.end()) {
							temp.push_back(-1.0);
						} else {
							temp.push_back(0.0);
						}
					}
				}
				M.push_back(temp);
			}
		}
		const vector<vector<double>> reveal() {
			return M;
		}
		int getrealsize() {
			return realsize;
		}

		// solve matching with |Query| cardinality as base
		double solveQ(int q, std::set<pair<double, int>, cmp_increasing> *L_, std::mutex *mtx, double *etm, bool isEarly = true){
			HungarianAlgorithm HungAlgo;
			vector<int> assignment;
			double cost = HungAlgo.Solve(M, assignment, q, L_, mtx, etm, isEarly);
			double overlap = 0.0;
			if (matching == 0) {
				return 0.0;
			}
			return -cost/q;
		}

};

/********************************
  GPU FAISS Index Wrapper
 ********************************/
class FaissIndexGpu {

	private:
		faiss::gpu::GpuIndexFlatIP *index;
		unordered_map<int, vector<float>> normalized;
		vector<int> dictionary;

	public:
		FaissIndexGpu(string path, Database *db, unordered_set<int> validset){
			string indexpath = path + "faiss.index";
			int d = 300;

			vector<vector<float>> vectors = db->get_valid_vectors(validset);
			dictionary = db->get_dictionary();

			index = new faiss::gpu::GpuIndexFlatIP(&res, d);
			
			int nb = vectors.size();
			float *xb = new float[d * nb];
			for (int i = 0; i < nb; i++){
				for (int j = 0; j < d; j++){
					xb[d * i + j] = vectors[i][j];
				}
			}

			faiss::fvec_renorm_L2(d, nb, xb);

			for (int i = 0; i < nb; i++){
				vector<float> vec;
				for (int j = 0; j < d; j++){
					vec.push_back(xb[d * i + j]);
				}
				normalized[dictionary[i]] = vec;
			}
			printf("%s\n", "Normalized vector storage check");

			cout << endl;
			index->add(nb, xb);
			int k = 4;

		}

		tuple<vector<idx_t>, vector<float>> search(int nq, vector<float> vxq, int k){

			int d = 300;

			float* xq = vxq.data();

			// search xq
			idx_t *I = new idx_t[k * nq];
			float *D = new float[k * nq];

			index->search(nq, xq, k, D, I);

			vector<idx_t> vI(I, I + k * nq);
			vector<float> vD(D, D + k * nq);

			delete [] I;
			delete [] D;
			return {vI, vD};
		}

		vector<float> get_normalized_vector(int token){
			return normalized[token];
		}

		vector<int> get_dictionary(){
			return dictionary;
		}
		void destroy(){
			index = NULL;
			cout << "Destroyed Faiss Index" << endl;
		}


};


/********************************
 * Bucket Upperbound
 ********************************/
class BucketUpperBound{
private:
    int q;
	struct cmp {
		bool operator() (pair<double, int> lhs, pair<double, int> rhs) const {
			return std::get<0>(lhs) > std::get<0>(rhs);
		}
	};

	unordered_map<int, std::set<pair<double, int>, cmp>> buckets;
	unordered_map<int, double> siVal;
	unordered_map<int, int> ciAt;

public:
    BucketUpperBound(int cardinality){
        q = cardinality;
    }

	void initialize(int c_id, std::set<int> intersection, int cand_cardinality) {
		double si = static_cast<double>(intersection.size());
		int m = min(q, cand_cardinality) - si;
		siVal[c_id] = si;
		ciAt[c_id] = m;
		buckets[m].insert(make_pair(si, c_id));
	}
		// update bucket upperbound with input edge
		// update(candidate_set, similarity, candidate_set_cardinality)
    double update(int ci, double s, int t) {
		// cout << "Enter bucket update: " << ci << endl;
		double si = 0.0;
		int m = 0;
        if (siVal.find(ci) != siVal.end()){
			// cout << "Found ci" << endl;
			si = siVal[ci];
			m = ciAt[ci];
			// cout << "Before: " << si << " " << m << " " << s << " " << endl;
			if (m < 1) {
				return si / q;
			}
			auto it = buckets[m].find(make_pair(si, ci));
			if (it != buckets[m].end()){
				buckets[m].erase(it);
			}
			si += s;
			m -= 1;
			// cout << "After: " << si << " " << m << " " << s << " " << endl;
			siVal[ci] = si;
			buckets[m].insert(make_pair(si, ci));
			ciAt[ci] = m;
		} else {
			// cout << "init ci" << endl;
			si = s;
			m = min(q, t) - 1;
			// cout << si << " " << m << " " << s << " " << endl;
			siVal[ci] = si;
			ciAt[ci] = m;
			buckets[m].insert(make_pair(si, ci));
		} 
		
		// cout << (si + m * s) / q << endl;
		return (si + m * s) / q;
    }

	// prune bucket with last seen smallest cosine similarity
	vector<int> prune(double s, double thetaK) {
		// cout << "Enter bucket pruning: " << "ThetaK = " << thetaK << " | " << "s = " << s <<endl;
		vector<int> pruned;
		for (const auto & [key, value] : buckets) {
			int m = key;
			auto temp = buckets[m].begin();
			double si = std::get<0>(*temp);
			double ci = std::get<1>(*temp);
			while (si <= thetaK * q - m * s) {
				// cout << si << " " << ci << endl;
				if (temp != buckets[m].end()){
					buckets[m].erase(temp);
					siVal.erase(ci);
					ciAt.erase(ci);
					pruned.push_back(ci);
				} else {
					break;
				}
				temp = buckets[m].begin();
				si = std::get<0>(*temp);
				ci = std::get<1>(*temp);
			}
		}
		return pruned;
	}
	// get(set_id, 0.0)
	double get(int ci, double s, bool debug = false) {
		int m = ciAt[ci];
		double si = siVal[ci];
		return (si + m * s) / q;
	}

	void set_upperbound(int ci, double sim) {
		siVal[ci] = sim;
	}

    

};

/********************************
 * Refined Lowerbound
 ********************************/
// Basically, greedy matching built up by streaming in edges
// But refined by validedges gathered during posting list popping in mind
// The remaining value in matchings below alpha are considered 0
class RefinedLowerBound {
	private:
		int q;
		unordered_map<int, set<int>> mapped_query_tokens; // key: candidate_set --> value : mapped query tokens
		unordered_map<int, set<int>> mapped_cand_tokens; // key: candidate_set --> value : mapped candidate tokens
		unordered_map<int, double> candidate_set_si;

	public:
		RefinedLowerBound(int cardinality){
			q = cardinality;
		}

		void initialize_lower_bound(int c_id, std::set<int> intersection) {
			candidate_set_si.insert({c_id, static_cast<double>(intersection.size())});
			mapped_cand_tokens.insert({c_id, intersection});
			mapped_query_tokens.insert({c_id, intersection});
		}
		// update(candidate_set_id, similairty, query_token, candidate_token)
		double update(int c_id, int sim, int qtoken, int ttoken, set<int> query_tokens, set<int> candidate_tokens){
			auto it = mapped_cand_tokens.find(c_id);
			bool ismappedqt = std::find(mapped_query_tokens.at(c_id).begin(), mapped_query_tokens.at(c_id).end(), qtoken) != mapped_query_tokens.at(c_id).end();
			bool ismappedct = std::find(mapped_cand_tokens.at(c_id).begin(), mapped_cand_tokens.at(c_id).end(), ttoken) != mapped_cand_tokens.at(c_id).end();
			if (!ismappedqt && !ismappedct) {
				mapped_query_tokens.at(c_id).insert(qtoken);
				mapped_cand_tokens.at(c_id).insert(ttoken);
				candidate_set_si.at(c_id) += sim;
			}
			return candidate_set_si.at(c_id) / q;
		}

		double get(int c_id, set<int> query_tokens, set<int> candidate_tokens) {
			return candidate_set_si.at(c_id) / q;
		}

		void set_lowerbound(int c_id, double new_sim) {
			candidate_set_si.at(c_id) = new_sim;
		}
};

/********************************
 * matching function to be parallelized 
 ********************************/
double thread_helper_nocache( 
	set<int> query_tokens, 
	set<int> target_tokens, 
	unordered_map<size_t, double> validedge,
 	std::set<pair<double, int>, cmp_increasing> *tkLB,
	std::mutex *mtx,
	double *etm){
		double similarity = 0.0;
		ValidMatrix *m = new ValidMatrix(query_tokens, target_tokens, validedge);
		similarity = m->solveQ(query_tokens.size(), tkLB, mtx, etm);
		return similarity;
}




class ParallelBoundsUpdate {
	private:
		vector<int> posting_list;
		std::set<int> *candidate_sets;
		std::set<int> *pruned;
		set<int> *query_tokens;
		unordered_map<int, set<int>> *sets;
		Database *ftdb;
		int tbest;
		int tqbest;
		double sim;
		double alpha;
		int query_i;
		std::set<pair<double, int>, cmp_increasing> *tkLB;
		BucketUpperBound *bucket_upperbound;
		RefinedLowerBound *lower_bounds;
		std::mutex *mtx;
		double *considered;
		double *upperpruned;
		double *top_one_count;
		std::set<int> *top_one_sets;
		double *global_thetaK;
		unordered_map<size_t, double> validedge;
		double *etm;
		int k;
	public:
		void operator()(const oneapi::tbb::blocked_range<size_t>& range) const {
			for (size_t i = range.begin(); i < range.end(); ++i) {
				/**
				combine the upperbound and lowerbound
				*/
				int x = posting_list[i];
				

				mtx->lock();
				bool iscandidate = std::find(candidate_sets->begin(), candidate_sets->end(), x) != candidate_sets->end();
				bool ispruned = std::find(pruned->begin(), pruned->end(), x) != pruned->end();
				mtx->unlock();
				if (!iscandidate && !ispruned && sim >= *global_thetaK) {
					mtx->lock();
					candidate_sets->insert(x);
					*considered += 1;
					double UBx = bucket_upperbound->update(x, sim, (*sets)[x].size());
					if (tkLB->size() < k) {
						set<int> qset = (*(sets->find(query_i))).second;
						set<int> tset = (*(sets->find(x))).second;
						double sim = thread_helper_nocache(qset, tset, validedge, tkLB, mtx, etm);
						tkLB->insert(make_pair(sim, x));
						bucket_upperbound->set_upperbound(x, sim);
					}
					mtx->unlock();
					iscandidate = true;
				}

			}
		}
		ParallelBoundsUpdate(
			vector<int> sets_cons,
			std::set<int> *cs,
			std::set<int> *p,
			set<int> *qt,
			unordered_map<int, set<int>> *s,
			Database *db,
			int tb,
			int tq,
			double simi,
			double a,
			int qi,
			std::set<pair<double, int>, cmp_increasing> *tklb,
			BucketUpperBound *bucket_ub,
			RefinedLowerBound *lbounds,
			std::mutex *m,
			double *con,
			double *upper,
			double *top_one,
			std::set<int> *top_one_s,
			double *global_th,
			unordered_map<size_t, double> ve,
			double *e,
			int k_
		) {
			posting_list = sets_cons;
			candidate_sets = cs;
			pruned = p;
			query_tokens = qt;
			sets = s;
			ftdb = db;
			tbest = tb;
			tqbest = tq;
			sim = simi;
			alpha = a;
			query_i = qi;
			tkLB = tklb;
			bucket_upperbound = bucket_ub;
			lower_bounds = lbounds;
			mtx = m;
			considered = con;
			upperpruned = upper;
			top_one_count = top_one;
			top_one_sets = top_one_s;
			global_thetaK = global_th;
			validedge = ve;
			etm = e;
			k = k_;
		}

};

class BackgroundCalculator {
	private:
		std::vector<int> being_computed;
		std::set<pair<double, int>, cmp_increasing> *tkUB;
		std::set<pair<double, int>, cmp_increasing> *tkLB;
		std::set<pair<double, int>, cmp_decreasing> *upperbound_pq;
		std::unordered_map<int, double> *computed_exact;
		std::unordered_map<int, bool> *checked;
		std::set<int> query_tokens;
		std::unordered_map<int, set<int>> sets;
		RefinedLowerBound *lowerbounds;
		BucketUpperBound *buckets;
		Database *ftdb;
		std::mutex *mtx;
		double alpha;
		double *early_pruned;
		int k;
		unordered_map<size_t, double> validedge;
		int query_i;
		double *etm;
	public:
		void start(thread_pool *pool) {
			// the last set is being passed to all threads, why????
			// as thread need reference not pointers or variables itself.
			for (auto& target_set : being_computed) {
				// cout << "set: " << target_set << endl;
				pool->push_task([&, this]{ semantic_overlap(target_set);});
			}
		}

		void post_processing_phase() {
			thread_pool pool(std::thread::hardware_concurrency() - 13);
			start(&pool);
			int target_set = get_next_set(tkUB, checked, computed_exact);
			int counter = 0;
			while (!all_checked(tkUB, checked)) {
				if (target_set != -1) {
					// mtx->lock();
					checked->at(target_set) = true;
					double sim = computed_exact->at(target_set);
					// lowerbounds->set_lowerbound(target_set, sim);
					auto it = std::find_if(tkUB->begin(), tkUB->end(), [target_set](const pair<double, int>& p){ return p.second == target_set;}); // find the target set in the topK-UB
					tkUB->erase(it); // remove this set from topK UB
					if (sim >= tkUB->begin()->first) {
						tkUB->insert(make_pair(sim, target_set));
					}
					if (sim >= tkLB->begin()->first) {
						auto it = std::find_if(tkLB->begin(), tkLB->end(), [target_set](const pair<double, int>& p){ return p.second == target_set;}); // find the target set in the topK-UB
						if (it != tkLB->end()) {
							tkLB->erase(it);
							tkLB->insert(make_pair(sim, target_set));
						} else {
							tkLB->insert(make_pair(sim, target_set));
							tkLB->erase(tkLB->begin());
						}
						upperbound_pq->insert(make_pair(sim, target_set));
					}
					while (tkUB->size() < k && upperbound_pq->size() > 0) {
						if (upperbound_pq->begin()->first >= tkLB->begin()->first) {
							std::pair<double, int> top_pq = *(upperbound_pq->begin());
							tkUB->insert(top_pq);
							if (!checked->at(top_pq.second)) {
								push_new_set(&pool, std::ref(top_pq.second));
							}
							upperbound_pq->erase(upperbound_pq->begin());
						} 
					}
				} else {
					counter += 1;
				}
				if (counter % 10 == 0) {
					pool.wait_for_tasks();
				}
				target_set = get_next_set(tkUB, checked, computed_exact);
			}
			pool.paused = true;
		}

		void push_new_set(thread_pool *pool, int& target_set) {
			pool->push_task([&, this]{ semantic_overlap(target_set);});
		}

		void semantic_overlap(int target_set) {
			if (computed_exact->find(target_set) == computed_exact->end()) {
				set<int> qset = (*(sets.find(query_i))).second;
				set<int> tset = (*(sets.find(target_set))).second;
				double sim = thread_helper_nocache(qset, tset, validedge, tkLB, mtx, etm);
				mtx->lock();
				computed_exact->insert({target_set, sim});
				mtx->unlock();
				
			}
		}

		int get_next_set(std::set<pair<double, int>, cmp_increasing> *tkub, std::unordered_map<int, bool> *ch, std::unordered_map<int, double> *ce) {
			for (auto it = tkub->begin(); it != tkub->end(); ++it) {
				int set = it->second;
				if (!ch->at(set) && ce->find(set) != ce->end()) {
					return set;
				}
			}
			return -1;

		}

		bool all_checked(std::set<pair<double, int>, cmp_increasing> *tkub, std::unordered_map<int, bool> *ch) {
			for (auto it = tkub->begin(); it != tkub->end(); ++it) {
				if (!ch->at(it->second)) {
					return false;
				}
			}
			return true;
		}

		BackgroundCalculator(
			std::vector<int> being_c,
			std::set<pair<double, int>, cmp_increasing> *tUB,
			std::set<pair<double, int>, cmp_increasing> *tLB,
			std::set<pair<double, int>, cmp_decreasing> *pq,
			std::unordered_map<int, double> *cexact,
			std::unordered_map<int, bool> *check,
			std::set<int> qt,
			std::unordered_map<int, set<int>> s,
			RefinedLowerBound *lbs,
			BucketUpperBound *buck,
			Database *db,
			std::mutex *m,
			double a,
			double *ep,
			int K,
			unordered_map<size_t, double> vedge,
			int qi,
			double *early_term
		) {
			being_computed = being_c;
			tkUB = tUB;
			tkLB = tLB;
			upperbound_pq = pq;
			computed_exact = cexact;
			checked = check;
			query_tokens = qt;
			sets = s;
			lowerbounds = lbs;
			buckets = buck;
			ftdb = db;
			mtx = m;
			alpha = a;
			early_pruned = ep;
			k = K;
			validedge = vedge;
			query_i = qi;
			etm = early_term;
		}
};

bool theta_prunable(std::set<pair<double, int>, cmp_increasing> *tkLB) {
	for (auto it = tkLB->begin(); it != tkLB->end(); it++) {
		if (it->second == -1) {
			return false;
		}
	}
	return true;
}

/********************************
 ALGORITHM
 ********************************/
void algorithm(Environment *env, Database *ftdb, FaissIndexGpu *faissindex, string query, int k, int size, double alpha,
						double *global_considered, double *global_upperpruned, double *globalpretime, double *globalposttime, 
						double *globaltopone, double *globalpostsets, double *globaltokenstreamsize, std::set<pair<double, int>, cmp_decreasing> *finalresults, 
						vector<bool> *status_tracker, std::mutex *global_gmt, int *global_query_i, 
						double *global_query_card, double *global_thetaK, double *global_early_pp, 
						double *global_early_termination, double *global_results_found, double *response_time_pre, double *response_time_post) {
	
	unordered_map<int, set<int>> sets = env->getSets(); // all sets stored as key: set integer id, value: set data (int token integer ids)
	unordered_set<int> wordset = env->getWordset(); // all unique tokens in copora
	vector<set<int>> inverted_index = env->getRindex(); // inverted index that returns all sets containing given token
	vector<int> dictionary = ftdb->get_dictionary(); // fastText database instance
	std::mutex gmtx_internal; //  mutex lock

	// benchmarking metrics
	double upperpruned = 0;
	double considered = 0;
	double postprocessing_sets = 0;
	double preprocessing_time = 0.0;
	double postprocessing_time = 0.0;
	double early_pruned_post_processing = 0;
	double top_one_LB_sets_found = 0;
	double token_stream_size = 0;
	double early_termination_match = 0;

	int query_i = env->getSetId(query); // get setID for the query
	std::set<int> query_tokens = sets[query_i];
	std::set<int> candidate_sets; // candidate sets that can't be pruned by bounds
	std::unordered_map<int, int> token_stream_tracker; //keep track of each posting list's last pop location
	std::set<int> pruned; //sets once pruned from candidate cannot come back in
	std::set<pair<double, int>, cmp_increasing> tkLB; // Top-K LB
	std::set<pair<double, int>, cmp_increasing> tkUB; // Top-K UB
	std::set<int> top_one_sets; // keep track of sets that have LB = 1
	priority_queue<ptuple> token_stream; // queue contains edges from posting lists in decreasing order
	unordered_map<size_t, double> validedge; // valid edge that we do not need to calculate cosine similarity again
	double thetaK = 0.0; // thetaK tracker
	// double max_LB_seen = 0.0; // to help binary search optimal thetaK
	unordered_map<int, vector<float>> token_to_norm_vector;
	// posting list trackers
	int tbest = 0; // current best candidate token
	int tqbest = 0; // curent best token's corresponding query token
	BucketUpperBound *bucket_upperbound = new BucketUpperBound(query_tokens.size());
	RefinedLowerBound *lowerbounds = new RefinedLowerBound(query_tokens.size());

	// ************************** INITIALIZATION PHASE ******************************
	std::chrono::time_point<std::chrono::high_resolution_clock> prestart, preend; // preprocessing time trackers
	std::chrono::duration<double> prelapsed;
	prestart = std::chrono::high_resolution_clock::now();

	
	// track how many tokens where read for the query token
	for (auto it = query_tokens.begin(); it != query_tokens.end(); it++){
		token_stream_tracker[*it] = 0;
	}


	int i = 0;
	vector<float> vxq;
	int nq = 0;
	for (auto it = query_tokens.begin(); it != query_tokens.end(); it++){
		int tq = *it;
		if (std::find(dictionary.begin(), dictionary.end(), tq) == dictionary.end()){
			// not semantic
			token_stream.push(make_pair(1.0, make_pair(tq, tq)));
		} else {
			// semantic
			vector<float> vec = ftdb->get_normalized_vector(tq);
			token_to_norm_vector.insert(make_pair(tq, vec)); // cache
			if (vec.size() == 0){
				cerr << "Vector should not be empty" << endl;
			}
			vxq.insert(vxq.end(), vec.begin(), vec.end());
			nq += 1;
		}
		i += 1;
	}

	// ************************** first scan of posting list ******************************
	int scanK = 100; // each time populate posting list with depth 100
	// cout << "VXQ: " << vxq.size() << endl;
	global_gmt->lock();
	tuple<vector<idx_t>, vector<float>> rt = faissindex->search(nq, vxq, scanK);
	global_gmt->unlock();
	vector<idx_t> I = std::get<0>(rt);
	vector<float> D = std::get<1>(rt);
	int cur = 0;
	for (vector<idx_t>::iterator it = I.begin(); it != I.end(); it++) {
		int tq_cur = I[cur - (cur % scanK)];
		int tq = dictionary[tq_cur];
		int word = dictionary[*it];
		float fsim = D[cur];
		double sim = static_cast<double>(fsim);
		if (sim >= alpha) {
			token_stream.push(make_pair(sim, make_pair(word, tq)));
		}
		cur += 1;
	}

	
	// ************************** REFINEMENT PHASE ******************************
	// ************************** master loop ******************************
	double sim = 1.0;
	// double thetaK = 0.9;
	// tkLB.insert(make_pair(thetaK, -1));
	token_stream_size = token_stream.size();
	while (token_stream.size() > 0) {
		ptuple temp = token_stream.top();
		token_stream.pop();
		tbest = std::get<0>(std::get<1>(temp));
		tqbest = std::get<1>(std::get<1>(temp));
		sim = std::get<0>(temp);
		// faiss returns 0.9999 for similar tokens
		if (tbest == tqbest) {
			sim = 1.0;
		}
		// skip over sets when tbest == tqbest and initialize the upper bound with overlap score. 
		validedge[key(tqbest, tbest)] = sim;
		token_stream_tracker[tqbest] += 1;
		
		double old_thetaK = tkLB.begin()->first;
		std::set<int> sets_considered = inverted_index[tbest];
		std::set<int> intersection;
		std::vector<int> sets_considered_vector(sets_considered.begin(), sets_considered.end());

		// cout << "Arrives at Parallel Bounds Update" << endl;
		ParallelBoundsUpdate *pbu = new ParallelBoundsUpdate(sets_considered_vector, &candidate_sets, &pruned, &query_tokens, &sets, ftdb, tbest, tqbest,
																sim, alpha, query_i, &tkLB, bucket_upperbound, lowerbounds, &gmtx_internal, &considered, 
																&upperpruned, &top_one_LB_sets_found, &top_one_sets, 
																global_thetaK, validedge, &early_termination_match, k);
		oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, sets_considered_vector.size()), (*pbu));
		// cout << "Leaves at Parallel Bounds Update" << endl;
		


		if (tkLB.begin()->first > *global_thetaK) {
			// cout << "Old Global ThetaK: " << *global_thetaK << endl;
			global_gmt->lock();
			*global_thetaK = tkLB.begin()->first;
			global_gmt->unlock();
			// cout << "New Global ThetaK: " << *global_thetaK << endl;
		}

		// step scan posting list when reach next K level
		if (token_stream_tracker[tqbest] % scanK == 0) {
			vxq = token_to_norm_vector.at(tqbest);
			global_gmt->lock();
			tuple<vector<idx_t>, vector<float>> rt = faissindex->search(1, vxq, token_stream_tracker[tqbest] + scanK);
			global_gmt->unlock();
			vector<idx_t> I = std::get<0>(rt);
			vector<float> D = std::get<1>(rt);
			int cur = 0;
			int inserted = 0;
			for (vector<idx_t>::iterator it = I.begin(); it != I.end(); it++){
				if (cur > token_stream_tracker[tqbest]) {
					int tq_cur = I[cur];
					int tq = dictionary[tq_cur];
					int word = dictionary[*it];
					float fsim = D[cur];
					double nsim = static_cast<double>(fsim);
					if (nsim >= alpha) {
						token_stream.push(make_pair(nsim, make_pair(word, tq)));
						inserted += 1;
					}
				}
				cur += 1;
			}
			// cout << "Inserted: " << inserted << endl;
			token_stream_size += inserted;
		}
	}
	preend = std::chrono::high_resolution_clock::now();
	prelapsed = preend - prestart;
	preprocessing_time = prelapsed.count();


	// ************************** POST PROCESSING PHASE ******************************
	std::chrono::time_point<std::chrono::high_resolution_clock> poststart, postend; // preprocessing time trackers
	std::chrono::duration<double> postlapsed;
	poststart = std::chrono::high_resolution_clock::now();
	std::set<pair<double, int>, cmp_decreasing> upperbound_pq; // upperbound priority queue
	std::unordered_map<int, bool> checked; // track whether a set has been checked or not.
	std::vector<int> being_computed;
	std::unordered_map<int, double> computed_exact;
	postprocessing_sets = candidate_sets.size();
	for (auto it = candidate_sets.begin(); it != candidate_sets.end(); it++) {
		int set = *it;
		upperbound_pq.insert(make_pair(0.0, set));
		checked.insert({set, false});
	}
	// initializing topK UB list
	for (auto it = upperbound_pq.begin(), next_it = it; it != upperbound_pq.end(); it = next_it) {
		++next_it;
		tkUB.insert(*it);
		being_computed.push_back(it->second);
		upperbound_pq.erase(it);
		if (tkUB.size() >= k) {
			break;
		}
	}
	// cout << "TopKUB Size: " << tkUB.size() << endl;

	BackgroundCalculator *bg_calc = new BackgroundCalculator(being_computed, &tkUB, &tkLB, &upperbound_pq, &computed_exact, &checked, query_tokens, sets, 
																lowerbounds, bucket_upperbound, ftdb, &gmtx_internal, 
																alpha, &early_pruned_post_processing, k, validedge, query_i, &early_termination_match); 

	// cout << "start post processing" << endl;
	bg_calc->post_processing_phase();
	postend = std::chrono::high_resolution_clock::now();
	postlapsed = postend - poststart;
	postprocessing_time = postlapsed.count();

	global_gmt->lock();
	*global_considered += considered;
	*global_upperpruned += upperpruned;
	*globalpostsets += postprocessing_sets;
	*globalpretime += preprocessing_time;
	*globalposttime += postprocessing_time;
	*response_time_pre = std::max(*response_time_pre, preprocessing_time);
	*response_time_post = std::max(*response_time_post, postprocessing_time);
	*globaltopone += top_one_LB_sets_found;
	*globaltokenstreamsize += token_stream_size;
	*global_query_i = query_i;
	*global_query_card = static_cast<double>(query_tokens.size());
	*global_early_pp += early_pruned_post_processing;
	*global_early_termination += early_termination_match;
	*global_results_found += static_cast<double>(tkUB.size());
	// status_tracker->at(partition_number) = true;
	global_gmt->unlock();
}

/***************************************** 
	Main method, program entry point
******************************************/

int main(int argc, char const *argv[]){
	// arguements
	string setloc = "/localdisk2/wdc/data_lake_70c/";
	// string setloc = "/localdisk2/cleansets70/";	
	string query = "";
	string writeto = "./tempp/";
	string writetok = "./resultsk/";
	int number_of_partitions = 10;
	int k = 10;
	vector<string> queries;
	double alpha = 0.75;
	bool isCost = false;
	bool isStrict = false;
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end; 
	std::chrono::duration<double> elapsed;

	// reading input parameters

	if (argc > 1) {
		query = argv[1];
	}
	if (argc > 2) {
		writeto = argv[2];
	}
	if (argc > 3) {
		alpha = stod(argv[5]);
	}

	double faissindextime= 0.0;
	double invertedindextime = 0.0;
	double inverted_index_size = 0;
	// setting up environment, sql_database and faissindex
	std::chrono::time_point<std::chrono::high_resolution_clock> envstart, envend, faisstart, faissend; // preprocessing time trackers
	std::chrono::duration<double> envlapsed, faisselapsed;
	envstart = std::chrono::high_resolution_clock::now();
	Environment *env = new Environment(setloc, 10000000);
	envend = std::chrono::high_resolution_clock::now();
	envlapsed = envend - envstart;
	invertedindextime = envlapsed.count();
	vector<set<int>> inverted_index = env->getRindex(); // inverted index that returns all sets containing given token
	for (set<int> f : inverted_index) {
		inverted_index_size += f.size();
	}
	inverted_index_size += inverted_index.size();
	cout << "Inverted Index Size: " << inverted_index_size << endl;
	Database *ftdb = new Database("ft.sqlite3", env);
	cout << "Words: " << env->getWordset().size() << endl;

	// FAISS INDEX
	faisstart = std::chrono::high_resolution_clock::now();
	FaissIndexGpu *faissindex = new FaissIndexGpu("./", ftdb, env->getWordset());
	faissend = std::chrono::high_resolution_clock::now();
	faisselapsed = faissend - faisstart;
	faissindextime = faisselapsed.count();
	
	cout << "-------------------------utility check over--------------------------------" << endl;
	
	// if we batch queries
	string delimiter = ",";
	string token;
	size_t pos = 0;
	if (query.find(delimiter) != string::npos){
		while ((pos = query.find(delimiter)) != string::npos) {
			token = query.substr(0, pos);
			query.erase(0, pos + delimiter.length());
			queries.push_back(token);
		}
	} else {
		queries.push_back(query);
	}
	unordered_map<int, set<int>> sets = env->getSets();
	int qnum = 1;
	for (vector<string>::iterator it = queries.begin(); it != queries.end(); it++) {
		cout << qnum << "/" << queries.size() << endl;
		string query_set = *it;
		//benchmarking stats
		double preprocessing_time = 0.0;
		double postprocessing_time = 0.0;
		double considered = 0;
		double upperpruned = 0;
		double postprocessing_sets = 0;
		double top_oneLB = 0;
		double token_stream_size = 0;
		double query_cardinality = 0;
		int query_i = 0;
		double global_thetaK = 0.0;
		double early_pruned_post_processing = 0;
		double early_termination_match = 0;
		double global_results_found = 0;
		double response_time_pre = 0.0;
		double response_time_post = 0.0;
				
		// final_results_list
		std::set<pair<double, int>, cmp_decreasing> final_results;

		// keep track of async calls
		std::vector<future<void>> parallel_algo_pool;

		// launch parallel runs
		// int k = 10;
		string path = setloc + query_set;
		std::mutex gmtx_parallel;

		algorithm(env, ftdb, faissindex, path, k, 10000000, alpha, 
													&considered, &upperpruned, &preprocessing_time, &postprocessing_time, &top_oneLB, &postprocessing_sets, &token_stream_size, 
													&final_results, &completed, &gmtx_parallel, &query_i, 
													&query_cardinality, &global_thetaK, &early_pruned_post_processing, 
													&early_termination_match, &global_results_found, &response_time_pre, &response_time_post);

		// post results
		vector<string> tags = {"inverted_index_time", "faissindex_time", "prepropressing_time", "postprocessing_time", "total_time",
								"inverted_index_size", "considered_sets", "upperpruned", "overlap_score_one-sets", "postprocessing_sets",
								"token_stream_size", "query_cardinality", "early_pruned_post_processing", 
								"early_termination_match", "number_of_sets_found", "response_time_pre_processing", "response_time_post_processing", "response_time"};
		vector<double> results = {invertedindextime, faissindextime, preprocessing_time, postprocessing_time, preprocessing_time + postprocessing_time, inverted_index_size,
									considered, upperpruned, top_oneLB, postprocessing_sets, token_stream_size, 
									query_cardinality, early_pruned_post_processing, early_termination_match, global_results_found, 
									response_time_pre, response_time_post, response_time_post + response_time_pre};
		
		string writepath = writeto + query_set;
		ofstream OutFilew(writepath);
		cout << "Result for " << query_set << endl;
		for (int i = 0; i < results.size(); i++) {
			OutFilew << tags[i] << ": " << results[i] << endl;
			cout << tags[i] << ": " << results[i] << endl;
		}

		OutFilew.close();
		ftdb->clear_cache();
		qnum += 1;
	}
	ftdb->terminate();
	faissindex->destroy();
	return 0;
}
