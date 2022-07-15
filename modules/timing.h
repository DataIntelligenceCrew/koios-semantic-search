#ifndef TOPKSTREAM_TIMING_H
#define TOPKSTREAM_TIMING_H

// The MIT License (MIT)
// Copyright (c) 2018 Willi Mann
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <sys/time.h>
#include <time.h>
#include <sys/resource.h>
#include <climits>
#include <vector>
#include <string>

class Timing {
	public:
		struct Interval {
			std::string descriptor;
#ifdef GETRUSAGE
			struct timeval usertime;
			struct timeval usertime_start;
			struct timeval systemtime;
			struct timeval systemtime_start;
#else
			struct timespec time;
			struct timespec time_start;
#endif
			Interval(const std::string & descriptor);
			double getfloat() const;
			void start();
			void stop();
			void reset() noexcept;
		};
	protected:
		std::vector<Interval*> intervals;

	public:
		Interval * create_enroll(const std::string & descriptor);
		void enroll(Interval * interval);
		~Timing();
		friend std::ostream & operator<<(std::ostream & os, const Timing & timing);

};

#endif
