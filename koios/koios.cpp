/*
* KOIOS : Semantic Overlap C++ Implementation
* See LICENSE file 
*/

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

/**
* ENVIRONMENT VARIABLES
*/
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

/**
* SQLITE3 Utilities
*/
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

/**
* BIPARTITE GRAPH MATRIX
*/
class ValidMatrix {
	private:
		vector<vector<double>> M; // the matrix
		vector<vector<pair<int, int>>> M_tokens; // the matrix of tokens
		vector<int> non_exact_token_indices;
		vector<int> assignment_internal;
		int matching;
		int realsize;
	public:
		// building a valid matrix with query set and target set and valid edges valid edges stand 
        // for edges popped from the monotonically decreasing priority queue by edge weights 
        // (cosine similarity) by leveraging valid edges, we save computing resources by do not 
        // have to re-calculate cosine similarity for edges retrieved from the Faiss index (which is expensive)
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
			// vector<int> query_tokens;
			for(set<int>::iterator itq = query_set_pruned.begin(); itq != query_set_pruned.end(); ++itq) {
				vector<double> temp;
				vector<pair<int, int>> temp_tokens;
				int qword = *itq;
				for(set<int>::iterator itt = target_set_pruned.begin(); itt != target_set_pruned.end(); ++itt) {
					int tword = *itt;
					if (validedge.find(key(qword, tword)) != validedge.end()) {
						temp.push_back(0.0 - validedge[key(qword, tword)]);
						// temp.push_back(validedge[key(qword, tword)]);
					} else {
						if (qword == tword) {
							temp.push_back(-1.0);
							// temp.push_back(1.0);
						} else {
							temp.push_back(0.0);
						}
					}
					temp_tokens.push_back(make_pair(qword, tword));
				}
				M.push_back(temp);
				M_tokens.push_back(temp_tokens);
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
			double cost = HungAlgo.Solve(M, assignment, non_exact_token_indices, q, L_, mtx, etm, isEarly);
			double overlap = 0.0;
			assignment_internal = assignment;
			if (matching == 0) {
				return 0.0;
			}
			return -cost/q;
		}

		vector<pair<pair<int, int>, double>> get_non_exact_tokens() {
			vector<pair<pair<int, int>, double>> non_exact_tokens;
			int nRows = M.size();
			int nCols = M[0].size();
			for (int i = 0; i < nRows; i++) {
				if (std::find(non_exact_token_indices.begin(), non_exact_token_indices.end(), i) != non_exact_token_indices.end()) {
					int j = assignment_internal[i];
					double asim = -M[i][j];
					pair<int, int> qt_pair = M_tokens[i][j];
					non_exact_tokens.push_back(make_pair(qt_pair, asim));
				}
			}
			return non_exact_tokens;
		}
};

/**
* GPU Faiss Index Wrapper
*/
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

/**
* Bucket Upperbound (iUB-Filter)
*/
class BucketUpperBound {
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
            double si = 0.0;
            int m = 0;
            if (siVal.find(ci) != siVal.end()) {
                si = siVal[ci];
                m = ciAt[ci];
                if (m < 1) {
                    return si / q;
                }
                auto it = buckets[m].find(make_pair(si, ci));
                if (it != buckets[m].end()){
                    buckets[m].erase(it);
                }
                si += s;
                m -= 1;
                siVal[ci] = si;
                buckets[m].insert(make_pair(si, ci));
                ciAt[ci] = m;
            } else {
                si = s;
                m = min(q, t) - 1;
                siVal[ci] = si;
                ciAt[ci] = m;
                buckets[m].insert(make_pair(si, ci));
            } 
            
            return (si + m * s) / q;
        }

        // prune bucket with last seen smallest cosine similarity
        vector<int> prune(double s, double thetaK) {
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

        double get(int ci, double s, bool debug = false) {
            int m = ciAt[ci];
            double si = siVal[ci];
            return (si + m * s) / q;
        }

        void set_upperbound(int ci, double sim) {
            siVal[ci] = sim;
        }
};

/**
* Refined Lowerbound (iLB)
* Basically, greedy matching built up by streaming in edges, but refined by validedges gathered during posting 
* list popping in mind. The remaining value in matchings below alpha are considered 0
*/
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




