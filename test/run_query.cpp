/*-----------------------------------------------------------------------------
 *  run_benchmark.cpp - A benchmark for SuccinctBitVector.hpp
 *
 *  Coding-Style:
 *      emacs) Mode: C, tab-width: 8, c-basic-offset: 8, indent-tabs-mode: nil
 *      vi) tabstop: 8, expandtab
 *
 *  Authors:
 *      Takeshi Yamamuro <linguin.m.s_at_gmail.com>
 *-----------------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <sys/time.h>

#include <cstdio>
#include <iostream>
#include <memory>

#include "SuccinctBitVector.hpp"

#include "cmdline.h"
#include "glog/logging.h"

using namespace std;
using namespace succinct::dense;

static double  __gettimeofday_sec(void);
static void __show_result(double t,
                int count, const char *msg, ...);

/* Timer resulution: ms */
#define SV_START_MS_TIMER       \
        {                       \
                double  temp = __gettimeofday_sec()
#define SV_STOP_MS_TIMER(t)     \
                t = __gettimeofday_sec() - temp;     \
        }

int 
main(int argc, char **argv)
{
        SuccinctBitVector       bv;
        cmdline::parser         p;

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
        for (int i = 0; i < bsz; i++) {
                if (i % 2 == 0) {
                        nbits++;
                        bv.set_bit(1, i);
                }
        }

        bv.build();

        /* Generate test data-set */
        std::shared_ptr<uint32_t> rkwk(
                        new uint32_t[nloop],
                        default_delete<uint32_t>());
        std::shared_ptr<uint32_t> stwk(
                        new uint32_t[nloop],
                        default_delete<uint32_t>());

        CHECK(bsz != 0 && nbits != 0);

        for (int i = 0; i < nloop; i++)
                (rkwk.get())[i] = rand() % bsz;

        for (int i = 0; i < nloop; i++)
                (stwk.get())[i] = rand() % nbits;

        /* Start benchmarking rank & select */
        {
                double  tm;

                /* A benchmark for rank */
                SV_START_MS_TIMER;
                for (int i = 0; i < nloop; i++)
                        bv.rank((rkwk.get())[i], 1);
                SV_STOP_MS_TIMER(tm);

                __show_result(tm, nloop, "--rank");

                /* A benchmark for select */
                SV_START_MS_TIMER;
                for (int i = 0; i < nloop; i++)
                        bv.select((stwk.get())[i], 1);
                SV_STOP_MS_TIMER(tm);

                __show_result(tm, nloop, "--select");
        }

        return EXIT_SUCCESS;
}

/* --- Intra functions ---*/

double 
__gettimeofday_sec()
{
        struct timeval tv;
        gettimeofday(&tv, NULL);

        return tv.tv_sec + (double)tv.tv_usec*1e-6;
}

void
__show_result(double t, int count, const char *msg, ...)
{
        if (msg != NULL) {
                va_list vargs;

                va_start(vargs, msg);
                vfprintf(stderr, msg, vargs);
                va_end(vargs);

                cout << endl;
        }

        cout << " Total Time: " << t << endl;
        cout << " Throughputs(query/sec): " << count / t << endl;
        cout << " Unit Speed(sec/query): " << t / count << endl;
}
