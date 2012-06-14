/*-----------------------------------------------------------------------------
 *  SuccinctBitVector.hpp - A x86/64 optimized rank/select dictionary
 *
 *  Coding-Style:
 *      emacs) Mode: C, tab-width: 8, c-basic-offset: 8, indent-tabs-mode: nil
 *      vi) tabstop: 8, expandtab
 *
 *  Authors:
 *      Takeshi Yamamuro <linguin.m.s_at_gmail.com>
 *-----------------------------------------------------------------------------
 */

#ifndef __SUCCINCTBITVECTOR_HPP__
#define __SUCCINCTBITVECTOR_HPP__

#ifndef __STDC_LIMIT_MACROS
 #define __STDC_LIMIT_MACROS
#endif

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <emmintrin.h>
#include <vector>
#include <memory>

#include "glog/logging.h"

#define NDEBUG

using namespace std;

namespace succinct {
namespace dense {

#if  defined(__GNUC__) && __GNUC_PREREQ(2, 2)
 #define        __USE_POSIX_MEMALIGN__
#endif 

/* Block size (32-bit environment) */
#define BSIZE   32

#define LEVEL1_NUM      256
#define LEVEL2_NUM      BSIZE

#define CACHELINE_SZ    16
#define SIMD_ALIGN      4

#define BYTE2DWORD(x)   ((x) >> 2)
#define DWORD2BYTE(x)   ((x) << 2)

/* Locate the position of blocks */
#define LOCATE_BPOS(pos)        ((pos + BSIZE - 1) / BSIZE)

/* Defined by BSIZE */
typedef uint32_t        block_t;

#ifdef __USE_SSE_POPCNT__
static uint32_t
popcount(block_t b) {
        uint32_t        x;

        __asm__("popcnt %1, %0;" :"=r"(x) :"r"(b));

        return x;
}
#else
static const uint8_t
popcountArray[] = {
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
        1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
        1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
        2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
        1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
        2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
        2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
        3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8
};

static uint32_t
popcount(block_t b) {
        return	popcountArray[(b >> 24) & 0xff] +
                        popcountArray[(b >> 16) & 0xff] +
                        popcountArray[(b >> 8) & 0xff] +
                        popcountArray[b & 0xff];
}
#endif /* __USE_SSE_POPCNT__ */

static uint8_t
selectPos8(uint32_t d, int r) {
        CHECK(r < 8);

        if (d == 0 && r == 0) return 0;

        uint32_t ret = 0;

        /* NOTE: A input for bsf MUST NOT be 0 */
        for (int i = 0; i < r + 1; i++, d ^= 1 << ret)
                __asm__("bsf %1, %0;" :"=r"(ret) :"r"(d));

        return ret;
}

static uint32_t
selectPos(block_t blk, uint32_t r) {
        CHECK(r < 32);

        uint32_t nblock = 0;
        uint32_t cnt = 0;

        while (nblock < 4) {
                cnt = popcount((blk >> nblock * 8) & 0xff);

                if (r < cnt) break;

                r -= cnt;
                nblock++;
        }

        return nblock * 8 +
                selectPos8((blk >> (nblock * 8)) & 0xff, r);
}

#ifdef __USE_POSIX_MEMALIGN__
static uint32_t *
cachealign_alloc(uint64_t size) {
        CHECK(size != 0);

        uint32_t        *p;
        uint32_t        ret;

        /* NOTE: *lev2 is 64B-aligned so as to avoid cache-misses */
        ret = posix_memalign((void **)&p,
                DWORD2BYTE(CACHELINE_SZ), DWORD2BYTE(size));

        return (ret == 0)? p : NULL;
}

static void
cachealign_free(uint32_t *p) {
        if (p == NULL) return;
        free(p);
}
#else
static uint32_t *
cachealign_alloc(uint64_t size) {
        CHECK(size != 0);

        uint32_t        *p;

        /* FIXME: *lev2 NEEDS to be 64B-aligned. */
        p = new uint32_t[size + CACHELINE_SZ];

        return p;
}

static void
cachealign_free(uint32_t *p) {
        if (p == NULL) return;
        delete[] p;
}
#endif /* __USE_POSIX_MEMALIGN__ */

class BitVector {
private:
        uint64_t        size; 
        uint64_t        none; 
        vector<block_t> B;

public:
        BitVector() : size(0), none(0) {}
        ~BitVector() throw() {}

        void init(uint64_t len) {
                CHECK(len != 0);

                size = len;

                B.reserve(LOCATE_BPOS(size));
                for (uint64_t i = 0; i <
                                LOCATE_BPOS(size); i++)
                        B.push_back(0);
        }

        void resize(uint64_t len) {
                CHECK(B.size() < LOCATE_BPOS(len));

                B.reserve(LOCATE_BPOS(len));
                for (uint64_t i = 0; i <
                                LOCATE_BPOS(len) - B.size(); i++)
                        B.push_back(0);

                size = len;
        }

        void set_bit(uint8_t bit, uint64_t pos) {
                CHECK(pos < size);

                if (bit)
                        B[pos / BSIZE] |= 1U << (pos % BSIZE);
                else
                        B[pos / BSIZE] &= (~(1U << (pos % BSIZE)));

                none++;
        }

        bool lookup(uint64_t pos) const {
                CHECK(pos < size);
                return (B[pos / BSIZE] & (0x01 << (pos % BSIZE))) > 0;
        }

        block_t get_block(uint64_t pos) const {
                CHECK(pos < size);
                return B[pos];
        }

        uint64_t length() const {
                return size;
        }

        uint64_t get_none() const {
                return none;
        }
}; /* BitVector */

class SuccinctRank;
class SuccinctSelect;

typedef std::shared_ptr<SuccinctRank>   RankPtr;
typedef std::shared_ptr<SuccinctSelect> SelectPtr;

class SuccinctRank {
private:
        uint32_t        size;
        std::shared_ptr<uint32_t>       mem;

        /*--- Private functions below ---*/
        void init(const BitVector& bv) {
                uint32_t *lev1p = NULL;
                uint8_t *lev2p = NULL;
                block_t *lev3p = NULL;

                uint32_t idx = 0, nbits = 0;

                do {
                        if (idx % LEVEL1_NUM == 0) {
                                lev1p = mem.get() + CACHELINE_SZ * (idx / LEVEL1_NUM);
                                lev2p = reinterpret_cast<uint8_t *>(lev1p + SIMD_ALIGN);
                                lev3p = reinterpret_cast<block_t *>(lev1p + SIMD_ALIGN + BYTE2DWORD(LEVEL1_NUM / LEVEL2_NUM));

                                *lev1p = nbits;
                        }

                        if (idx % LEVEL2_NUM == 0) {
                                CHECK(nbits - *lev1p <= UINT8_MAX);

                                *lev2p++ = static_cast<uint8_t>(nbits - *lev1p);
                                block_t blk = bv.get_block(idx / BSIZE);
                                memcpy(lev3p++, &blk, sizeof(block_t));
                        }

                        if (idx % BSIZE == 0)
                                nbits += popcount(bv.get_block(idx / BSIZE));
                } while (idx++ <= size);

                /* Put some tricky code here for SIMD instructions */
                for (uint32_t i = idx; i % LEVEL1_NUM != 0; i++) {
                        if (i % LEVEL2_NUM == 0)
                                *lev2p++ = static_cast<uint8_t>(UINT8_MAX);
                }
        };

        uint32_t rank1(uint32_t pos) const {
                CHECK(pos <= size);

                uint32_t *lev1p = mem.get() + CACHELINE_SZ * (pos / LEVEL1_NUM);
                uint8_t *lev2p = reinterpret_cast<uint8_t *>(lev1p + SIMD_ALIGN);

                uint32_t offset = (pos / LEVEL2_NUM) % (LEVEL1_NUM /LEVEL2_NUM);

                CHECK(offset < LEVEL1_NUM / LEVEL2_NUM);

                uint32_t r = *lev1p + *(lev2p + offset);

                block_t *blk = static_cast<block_t *>(lev1p + SIMD_ALIGN +
                                BYTE2DWORD(LEVEL1_NUM / LEVEL2_NUM)) + offset;
                uint32_t rem = (pos % LEVEL2_NUM) % BSIZE;

                r += popcount((*blk) & ((1ULL << rem) - 1));

                return r;
        }

public:
        SuccinctRank() : size(0) {};
        explicit SuccinctRank(const BitVector& bv) :
                size(bv.length()), mem(cachealign_alloc(CACHELINE_SZ *
                                ((bv.length() + LEVEL1_NUM - 1) / LEVEL1_NUM)),
                                cachealign_free) {init(bv);};
        ~SuccinctRank() throw() {};

        uint32_t rank(uint32_t pos, uint32_t bit) const {
                if (++pos > size)
                        throw "Overflow Exception: pos";

                if (bit)
                        return rank1(pos);
                else
                        return pos - rank1(pos);
        }

        std::shared_ptr<uint32_t> get_mem() const {
                return  mem;
        }
}; /* SuccinctRank */

class SuccinctSelect {
private:
        uint32_t        size;
        uint32_t        bit;
        RankPtr         rkQ;

        /*
         * A reference to the rank dictionary
         * of the orignal bit-vector.
         */
        RankPtr         rk;

        /*--- Private functions below ---*/
        void init(const BitVector& bv) {
                block_t         blk;
                BitVector       Q;

                uint64_t sz = bv.length();
                uint32_t bsize = (sz + BSIZE - 1) / BSIZE;

                /* Calculate sz in advance */
                for (uint32_t i = 0; i < bsize; i++) {
                        blk = block(bv.get_block(i), bit);

                        if ((i + 1) * BSIZE > sz) {
                                uint32_t rem = sz - i * BSIZE;
                                blk &= ((1 << rem) - 1);
                        }

                        size += popcount(blk);
                }

                Q.init(size);

                uint32_t qcount = 0;
                for (uint32_t i = 0; i < bsize; i++) {
                        if (i % (LEVEL1_NUM / BSIZE) == 0)
                                Q.set_bit(1, qcount);

                        blk = block(bv.get_block(i), bit);
                        qcount += popcount(blk);
                }

                RankPtr __rkQ(new SuccinctRank(Q));
                rkQ = __rkQ;
        }

        uint32_t cumltv(uint32_t val, uint32_t pos, uint8_t bit) const {
                if (bit)        return val;
                else    return pos - val;
        }

        block_t block(block_t b, uint8_t bit) const {
                if (bit)        return b;
                else    return ~b;
        }

public:
        SuccinctSelect() : size(0) {};
        explicit SuccinctSelect(const BitVector& bv,
                        uint8_t bit, RankPtr& srk) :
                size(0), bit(bit), rk(srk) {init(bv);};
        ~SuccinctSelect() throw() {};

        uint32_t select(uint32_t pos) const {
                CHECK(pos < size);

                /* Search the position on LEVEL1 */
                uint32_t lev1pos = rkQ->rank(pos, 1) - 1;

                uint32_t *lev1p = rk->get_mem().get() + CACHELINE_SZ * lev1pos;

                CHECK(pos >= cumltv(*lev1p, lev1pos * LEVEL1_NUM, bit));

                uint8_t lpos = pos - cumltv(
                        *lev1p, lev1pos * LEVEL1_NUM, bit);

                /* Search the position on LEVEL2 */
                uint8_t *lev2p = reinterpret_cast<uint8_t *>(lev1p + SIMD_ALIGN);

                uint32_t lev2pos = 0;

                do {
                        if (cumltv(*(lev2p + lev2pos + 1),
                                        (lev2pos + 1) * LEVEL2_NUM, bit) > lpos)
                                break;
                } while (++lev2pos < LEVEL1_NUM / LEVEL2_NUM - 1);

                /* Count the left bits */
                CHECK(lpos >= cumltv(*(lev2p + lev2pos), lev2pos * LEVEL2_NUM, bit));

                uint32_t rem = lpos - cumltv(*(lev2p + lev2pos), lev2pos * LEVEL2_NUM, bit);

                block_t *blk = static_cast<block_t *>(lev1p + SIMD_ALIGN +
                                BYTE2DWORD(LEVEL1_NUM / LEVEL2_NUM)) + lev2pos;

                return lev1pos * LEVEL1_NUM + lev2pos * LEVEL2_NUM +
                        selectPos(block(*blk, bit), rem);
        }
}; /* SuccinctSelect */

class SuccinctBitVector {
private:
        /* A sequence of bit-array */
        BitVector       bv;

        /* A rank/select dictionary for dense */
        RankPtr         rk;
        SelectPtr       st0;
        SelectPtr       st1;

public:
        SuccinctBitVector() : rk((SuccinctRank *)0),
                st0((SuccinctSelect *)0), st1((SuccinctSelect *)0) {};
        ~SuccinctBitVector() throw() {};

        /* Functions to initialize */
        void init(uint64_t size) {bv.init(size);}

        void build() {
                if (bv.length() == 0)
                        throw "Not initialized yet: bv";

                RankPtr __rk(new SuccinctRank(bv));
                SelectPtr __st0(new SuccinctSelect(bv, 0, __rk));
                SelectPtr __st1(new SuccinctSelect(bv, 1, __rk));

                rk = __rk, st0 = __st0, st1 = __st1;
        }

        void resize(uint64_t size) {bv.resize(size);}

        void set_bit(uint8_t bit, uint64_t pos) {
                if (pos >= bv.length())
                        throw "Invalid input: pos";
                if (bit > 1)
                        throw "Invalid input: bit";

                bv.set_bit(bit, pos);
        }

        bool lookup(uint64_t pos) const {
                if (pos >= bv.length())
                        throw "Invalid input: pos";

                return bv.lookup(pos);
        }

        /* Rank & Select operations */
        uint32_t rank(uint32_t pos, uint8_t bit) const {
                if (pos >= bv.length())
                        throw "Invalid input: pos";
                if (bit > 1)
                        throw "Invalid input: bit";

                return rk->rank(pos, bit);
        }

        uint32_t select(uint32_t pos, uint8_t bit) const {
                if (pos >= bv.get_none())
                        throw "Invalid input: pos";
                if (bit > 1)
                        throw "Invalid input: bit";

                if (bit)        return st1->select(pos);
                else    return st0->select(pos);
        }

        uint64_t get_size() const {
                return 0;
        }
}; /* SuccinctBitVector */

} /* dense */
} /* succinct */

#endif /* __SUCCINCTBITVECTOR_HPP__ */
