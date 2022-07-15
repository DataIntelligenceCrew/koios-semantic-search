///////////////////////////////////////////////////////////////////////////////
// Hungarian.h: Header file for Class HungarianAlgorithm.
// 
// This is a C++ wrapper with slight modification of a hungarian algorithm implementation by Markus Buehren.
// The original implementation is a few mex-functions for use in MATLAB, found here:
// http://www.mathworks.com/matlabcentral/fileexchange/6543-functions-for-the-rectangular-assignment-problem
// 
// Both this code and the orignal code are published under the BSD license.
// by Cong Ma, 2016
// 

#ifndef HUNGARIAN_H
#define HUNGARIAN_H

#include <iostream>
#include <vector>
#include <oneapi/tbb.h>
#include <set>

using namespace std;

struct cmp_increasing {
	bool operator() (pair<double, int> lhs, pair<double, int> rhs) const {
		return std::get<0>(lhs) <= std::get<0>(rhs);
	}
};


class HungarianAlgorithm
{
public:
	HungarianAlgorithm();
	~HungarianAlgorithm();
	double Solve(vector <vector<double> >& DistMatrix, vector<int>& Assignment, vector<int>& indices, int q, std::set<pair<double, int>, cmp_increasing> *L_, std::mutex *m, double *etm, bool isEarly);

private:
	void assignmentoptimal(int *assignment, vector<int>& indices, double *cost, double *distMatrix, int nOfRows, int nOfColumns, int q, std::set<pair<double, int>, cmp_increasing> *L_, double *etm);
	void buildassignmentvector(int *assignment, bool *starMatrix, int nOfRows, int nOfColumns);
	void computeassignmentcost(int *assignment, vector<int>& indices, double *cost, double *distMatrix, int nOfRows);
	void step2a(int *assignment, vector<int>& indices, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, double *distMatrixIn, double *cost, int q, std::set<pair<double, int>, cmp_increasing> *L_, double *etm);
	void step2b(int *assignment, vector<int>& indices, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, double *distMatrixIn, double *cost, int q, std::set<pair<double, int>, cmp_increasing> *L_, double *etm);
	void step3(int *assignment, vector<int>& indices, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, double *distMatrixIn, double *cost, int q, std::set<pair<double, int>, cmp_increasing> *L_, double *etm);
	void step4(int *assignment, vector<int>& indices, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col, double *distMatrixIn, double *cost, int q, std::set<pair<double, int>, cmp_increasing> *L_, double *etm);
	void step5(int *assignment, vector<int>& indices, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, double *distMatrixIn, double *cost, int q, std::set<pair<double, int>, cmp_increasing> *L_, double *etm);
};


#endif