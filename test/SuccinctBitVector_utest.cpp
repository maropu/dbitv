/*-----------------------------------------------------------------------------
 *  SuccinctBitVector_utest.cpp - A unit test for SuccinctBitVector.hpp
 *
 *  Coding-Style:
 *      emacs) Mode: C, tab-width: 8, c-basic-offset: 8, indent-tabs-mode: nil
 *      vi) tabstop: 8, expandtab
 *
 *  Authors:
 *      Takeshi Yamamuro <linguin.m.s_at_gmail.com>
 *-----------------------------------------------------------------------------
 */

#include <gtest/gtest.h>
#include "SuccinctBitVector.hpp"

#define BITV_SZ         1000LL

using namespace succinct::dense;

class SuccinctBVTest : public ::testing::Test {
public:
        SuccinctBitVector       bv;

        virtual void SetUp() {
                bv.init(BITV_SZ);

                for (uint64_t i = 0; i < BITV_SZ; i++)
                        if (i % 2 == 0) bv.set_bit(1, i);

                bv.build();
        }

        virtual void TearDown() {}
};

TEST_F(SuccinctBVTest, rank) {
        uint32_t nrank0 = 0;
        uint32_t nrank1 = 0;

        for (int i = 0; i < BITV_SZ; i++) {
                if (bv.lookup(i)) nrank1++;
                else nrank0++;

                EXPECT_EQ(nrank0, bv.rank(i, 0)) << "Position: " << i;
                EXPECT_EQ(nrank1, bv.rank(i, 1)) << "Position: " << i;
        }
}

TEST_F(SuccinctBVTest, select) {
        uint32_t        nrank0;
        uint32_t        nrank1;

        nrank0 = 0;
        nrank1 = 0;

        for (int i = 0; i < BITV_SZ; i++) {
                if (bv.lookup(i)) {
                        EXPECT_EQ(i, bv.select(nrank1, 1))
                                << "Position: " << i
                                << " nrank1: " << nrank1;

                        nrank1++;
                } else {
                        EXPECT_EQ(i, bv.select(nrank0, 0))
                                << "Position: " << i
                                << " nrank0: " << nrank0;

                        nrank0++;
                }
        }
}
