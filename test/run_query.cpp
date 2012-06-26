/*-----------------------------------------------------------------------------
 *  run_benchmark.cpp - A benchmark for SuccinctBitVector.hpp
 *
 *  Coding-Style: google-styleguide
 *      https://code.google.com/p/google-styleguide/
 *
 *  Copyright 2012 Takeshi Yamamuro <linguin.m.s_at_gmail.com>
 *-----------------------------------------------------------------------------
 */

#include <sys/time.h>
#include <cmdline.h>
#include <glog/logging.h>

#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <algorithm>
#include <vector>

#include "SuccinctBitVector.hpp"

namespace {

const double BIT_DENSITY = 0.50;
const size_t NTRIALS = 11;

class Timer {
 public:
  Timer() : base_(get_time()) {}
  ~Timer() throw() {  }

  double elapsed() const {
    return get_time() - base_;
  }

  void reset() {
    base_ = get_time();
  }

 private:
  double  base_;

  static double get_time() {
    struct timeval  tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + static_cast<double>(tv.tv_usec*1e-6);
  }

  Timer(const Timer &);
  Timer &operator=(const Timer &);
};

/* Generate a randomized value */
uint32_t __xor128() {
  static uint32_t x = 123456789;
  static uint32_t y = 362436069;
  static uint32_t z = 521288629;
  static uint32_t w = 88675123;

  uint32_t t = (x ^ (x << 11));
  x = y, y = z, z = w;

  return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)));
}

void __show_result(std::vector<double>& tv,
                   int count, const char *msg, ...) {
  if (msg != NULL) {
    va_list vargs;

    va_start(vargs, msg);
    vfprintf(stdout, msg, vargs);
    va_end(vargs);

    printf("\n");
  }

  /* Get the value of median */
  sort(tv.begin(), tv.end());
  double t = tv[tv.size() / 2];

  printf(" Total Time(count:%d): %lf\n", count, t);
  printf(" Throughputs(query/sec): %lf\n", count / t);
  printf(" Unit Speed(sec/query): %lfns\n", t * 1000000000 / count);
}

} /* namespace */

int main(int argc, char **argv) {
  succinct::dense::SuccinctBitVector  bv;
  cmdline::parser p;

  /* Parse a command line */
  p.add<int>("nloop", 'l', "loop num (1000-1000000000)",
             false, 10000000, cmdline::range(1000, 1000000000));
  p.add<int>("bitsz", 'b', "bit size (1000-1000000000))",
             false, 1000000, cmdline::range(1000, 1000000000));

  p.parse_check(argc, argv);

  google::InitGoogleLogging(argv[0]);
#ifndef NDEBUG
  google::LogToStderr();
#endif /* NDEBUG */

  int nloop = p.get<int>("nloop");
  int bsz = p.get<int>("bitsz");

  /* Generate a sequence of bits */
  bv.init(bsz);

  uint32_t nbits = 0;
  uint64_t thres = (1ULL << 32) * BIT_DENSITY;
  for (int i = 0; i < bsz; i++) {
    if (__xor128() < thres) {
      nbits++;
      bv.set_bit(i, 1);
    }
  }

  bv.build();

  /* Generate test data-set */
  std::shared_ptr<uint32_t> rkwk(new uint32_t[nloop],
                                 std::default_delete<uint32_t>());
  std::shared_ptr<uint32_t> stwk(new uint32_t[nloop],
                                 std::default_delete<uint32_t>());

  CHECK(bsz != 0 && nbits != 0);

  for (int i = 0; i < nloop; i++)
    (rkwk.get())[i] = __xor128() % bsz;

  for (int i = 0; i < nloop; i++)
    (stwk.get())[i] = __xor128() % nbits;

  /* Start benchmarking rank & select */
  {
    std::vector<double> rtv;
    std::vector<double> stv;

    /* A benchmark for rank */
    for (size_t i = 0; i < NTRIALS; i++) {
      Timer t;

      for (int j = 0; j < nloop; j++)
        bv.rank((rkwk.get())[j], 1);

      rtv.push_back(t.elapsed());
    }

    __show_result(rtv, nloop, "--rank");

    /* A benchmark for select */
    for (size_t i = 0; i < NTRIALS; i++) {
      Timer t;

      for (int j = 0; j < nloop; j++)
        bv.select((stwk.get())[i], 1);

      stv.push_back(t.elapsed());
    }

    __show_result(stv, nloop, "--select");
  }

  return EXIT_SUCCESS;
}
