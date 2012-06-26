/*-----------------------------------------------------------------------------
 *  SuccinctBitVector.hpp - A x86/64 optimized rank/select dictionary
 *
 *  Coding-Style: google-styleguide
 *      https://code.google.com/p/google-styleguide/
 *
 *  Copyright 2012 Takeshi Yamamuro <linguin.m.s_at_gmail.com>
 *-----------------------------------------------------------------------------
 */

#ifndef __SUCCINCTBITVECTOR_HPP__
#define __SUCCINCTBITVECTOR_HPP__

#ifndef __STDC_LIMIT_MACROS
  #define __STDC_LIMIT_MACROS
#endif

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <climits>
#include <vector>
#include <memory>

#include <nmmintrin.h>

#include "glog/logging.h"

#ifndef NDEBUG
  #define __assert(x) CHECK(x)
#else
  #define __assert(x)
#endif 

namespace succinct {
namespace dense {

/* namespace { */

/* Defined by BSIZE */
typedef uint64_t  block_t;

static const size_t BSIZE = 64;
static const size_t PRESUM_SZ = 128;
static const size_t CACHELINE_SZ = 64;

#ifdef __USE_SSE_POPCNT__
static uint64_t popcount64(block_t b) {
#ifdef __x86_64__
  return _mm_popcnt_u64(b);
#else
  return _mm_popcnt_u32((b >> 32) & 0xffffffff) +
      _mm_popcnt_u32(b & 0xffffffff);
#endif
}
#else
static const uint8_t popcountArray[] = {
  0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
  1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
  1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
  2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
  1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
  2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
  2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
  3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8
};

static uint64_t popcount64(block_t b) {
  return	popcountArray[(b >> 56) & 0xff] + popcountArray[(b >> 48) & 0xff] +
      popcountArray[(b >> 40) & 0xff] + popcountArray[(b >> 32) & 0xff] +
      popcountArray[(b >> 24) & 0xff] + popcountArray[(b >> 16) & 0xff] +
      popcountArray[(b >> 8) & 0xff] + popcountArray[b & 0xff];
}
#endif /* __USE_SSE_POPCNT__ */

static const uint8_t selectPos_[] = {
  8,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
  5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
  6,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
  5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
  7,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
  5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
  6,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
  5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
  8,8,8,1,8,2,2,1,8,3,3,1,3,2,2,1,8,4,4,1,4,2,2,1,4,3,3,1,3,2,2,1,
  8,5,5,1,5,2,2,1,5,3,3,1,3,2,2,1,5,4,4,1,4,2,2,1,4,3,3,1,3,2,2,1,
  8,6,6,1,6,2,2,1,6,3,3,1,3,2,2,1,6,4,4,1,4,2,2,1,4,3,3,1,3,2,2,1,
  6,5,5,1,5,2,2,1,5,3,3,1,3,2,2,1,5,4,4,1,4,2,2,1,4,3,3,1,3,2,2,1,
  8,7,7,1,7,2,2,1,7,3,3,1,3,2,2,1,7,4,4,1,4,2,2,1,4,3,3,1,3,2,2,1,
  7,5,5,1,5,2,2,1,5,3,3,1,3,2,2,1,5,4,4,1,4,2,2,1,4,3,3,1,3,2,2,1,
  7,6,6,1,6,2,2,1,6,3,3,1,3,2,2,1,6,4,4,1,4,2,2,1,4,3,3,1,3,2,2,1,
  6,5,5,1,5,2,2,1,5,3,3,1,3,2,2,1,5,4,4,1,4,2,2,1,4,3,3,1,3,2,2,1,
  8,8,8,8,8,8,8,2,8,8,8,3,8,3,3,2,8,8,8,4,8,4,4,2,8,4,4,3,4,3,3,2,
  8,8,8,5,8,5,5,2,8,5,5,3,5,3,3,2,8,5,5,4,5,4,4,2,5,4,4,3,4,3,3,2,
  8,8,8,6,8,6,6,2,8,6,6,3,6,3,3,2,8,6,6,4,6,4,4,2,6,4,4,3,4,3,3,2,
  8,6,6,5,6,5,5,2,6,5,5,3,5,3,3,2,6,5,5,4,5,4,4,2,5,4,4,3,4,3,3,2,
  8,8,8,7,8,7,7,2,8,7,7,3,7,3,3,2,8,7,7,4,7,4,4,2,7,4,4,3,4,3,3,2,
  8,7,7,5,7,5,5,2,7,5,5,3,5,3,3,2,7,5,5,4,5,4,4,2,5,4,4,3,4,3,3,2,
  8,7,7,6,7,6,6,2,7,6,6,3,6,3,3,2,7,6,6,4,6,4,4,2,6,4,4,3,4,3,3,2,
  7,6,6,5,6,5,5,2,6,5,5,3,5,3,3,2,6,5,5,4,5,4,4,2,5,4,4,3,4,3,3,2,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,3,8,8,8,8,8,8,8,4,8,8,8,4,8,4,4,3,
  8,8,8,8,8,8,8,5,8,8,8,5,8,5,5,3,8,8,8,5,8,5,5,4,8,5,5,4,5,4,4,3,
  8,8,8,8,8,8,8,6,8,8,8,6,8,6,6,3,8,8,8,6,8,6,6,4,8,6,6,4,6,4,4,3,
  8,8,8,6,8,6,6,5,8,6,6,5,6,5,5,3,8,6,6,5,6,5,5,4,6,5,5,4,5,4,4,3,
  8,8,8,8,8,8,8,7,8,8,8,7,8,7,7,3,8,8,8,7,8,7,7,4,8,7,7,4,7,4,4,3,
  8,8,8,7,8,7,7,5,8,7,7,5,7,5,5,3,8,7,7,5,7,5,5,4,7,5,5,4,5,4,4,3,
  8,8,8,7,8,7,7,6,8,7,7,6,7,6,6,3,8,7,7,6,7,6,6,4,7,6,6,4,6,4,4,3,
  8,7,7,6,7,6,6,5,7,6,6,5,6,5,5,3,7,6,6,5,6,5,5,4,6,5,5,4,5,4,4,3,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,4,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,5,8,8,8,8,8,8,8,5,8,8,8,5,8,5,5,4,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,6,8,8,8,8,8,8,8,6,8,8,8,6,8,6,6,4,
  8,8,8,8,8,8,8,6,8,8,8,6,8,6,6,5,8,8,8,6,8,6,6,5,8,6,6,5,6,5,5,4,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,8,8,8,8,8,8,8,7,8,8,8,7,8,7,7,4,
  8,8,8,8,8,8,8,7,8,8,8,7,8,7,7,5,8,8,8,7,8,7,7,5,8,7,7,5,7,5,5,4,
  8,8,8,8,8,8,8,7,8,8,8,7,8,7,7,6,8,8,8,7,8,7,7,6,8,7,7,6,7,6,6,4,
  8,8,8,7,8,7,7,6,8,7,7,6,7,6,6,5,8,7,7,6,7,6,6,5,7,6,6,5,6,5,5,4,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,5,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,6,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,6,8,8,8,8,8,8,8,6,8,8,8,6,8,6,6,5,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,8,8,8,8,8,8,8,7,8,8,8,7,8,7,7,5,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,8,8,8,8,8,8,8,7,8,8,8,7,8,7,7,6,
  8,8,8,8,8,8,8,7,8,8,8,7,8,7,7,6,8,8,8,7,8,7,7,6,8,7,7,6,7,6,6,5,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,6,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,8,8,8,8,8,8,8,7,8,8,8,7,8,7,7,6,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
  8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7
};

static uint64_t selectPos(block_t blk, uint64_t r) {
  __assert(r <= 32);

  uint64_t nblock = 0;
  uint64_t cnt = 0;

  while (nblock < 8) {
    cnt = popcount64(static_cast<uint64_t>(
            ((blk >> nblock * 8) & 0xff)));

    if (r < cnt)
      break;

    r -= cnt;
    nblock++;
  }

  return nblock * 8 +
      selectPos_[(r << 8) + ((blk >> (nblock * 8)) & 0xff)];
}

/*
 * FIXME: rBlock has 32-byte eachs so that its factor
 * is easily aligned to cache-lines. The container needs
 * to align data implicitly(e.g., AlignedVector).
 */
typedef struct {
  block_t   b0;
  block_t   b1;
  uint64_t  rk;
  uint64_t  b0sum;
} rBlock;

class BitVector {
 public:
  BitVector() : size_(0), none_(0) {}
  ~BitVector() throw() {}

  void init(uint64_t len) {
    __assert(len != 0);

    size_ = len;
    size_t bnum = (size_ + BSIZE - 1) / BSIZE;

    B_.resize(bnum);
    for (size_t i = 0; i < bnum; i++)
      B_[i] = 0;
  }

  void set_bit(uint64_t pos, uint8_t bit) {
    __assert(pos < size_);
    B_[pos / BSIZE] |= uint64_t(1) << (pos % BSIZE);
    none_++;
  }

  bool lookup(uint64_t pos) const {
    __assert(pos < size_);
    return (B_[pos / BSIZE] & (uint64_t(1) << (pos % BSIZE))) > 0;
  }

  const block_t get_block(uint64_t pos) const {
    __assert(pos < size_);
    return B_[pos];
  }

  uint64_t length() const {
    return size_;
  }

  uint64_t bsize() const {
    return B_.size();
  }

  uint64_t get_none() const {
    return none_;
  }

 private:
  uint64_t  size_; 
  uint64_t  none_; 
  std::vector<block_t>  B_;
}; /* BitVector */

class SuccinctRank;
class SuccinctSelect;

typedef std::shared_ptr<SuccinctRank>   RankPtr;
typedef std::shared_ptr<SuccinctSelect> SelectPtr;

class SuccinctRank {
 public:
  SuccinctRank() : size_(0) {};
  explicit SuccinctRank(const BitVector& bv) :
      size_(bv.length()) {init(bv);};
  ~SuccinctRank() throw() {};

  uint64_t rank(uint64_t pos, uint8_t bit) const {
    __assert(pos < size_);

    pos++;

    if (bit)
      return rank1(pos);
    else
      return pos - rank1(pos);
  }

  const rBlock& get_rblock(uint64_t idx) const {
    return  rblk_[idx];
  }

  rBlock& get_rblock(uint64_t idx) {
    return  rblk_[idx];
  }

 private:
  /*--- Private functions below ---*/
  void init(const BitVector& bv) {
    size_t bnum = bv.length() / PRESUM_SZ + 1;
    rblk_.resize(bnum);

    uint64_t r = 0;
    size_t pos = 0;
    for (size_t i = 0; i < bnum; i++, pos += 2) {
      uint64_t b0 = (pos < bv.bsize())? bv.get_block(pos) : 0;
      uint64_t b1 = (pos + 1 < bv.bsize())? bv.get_block(pos + 1) : 0;
      rblk_[i].b0 = b0;
      rblk_[i].b1 = b1;
      rblk_[i].rk = r;

      /* b0sum used for select() */
      uint64_t b0sum = popcount64(b0);
      rblk_[i].b0sum = b0sum;

      r += b0sum;
      r += popcount64(b1);
    }
  }

  uint64_t rank1(uint64_t pos) const {
    __assert(pos <= size_);

    const rBlock& rblk = rblk_[pos / PRESUM_SZ];

    uint64_t ret = rblk.rk;
    uint64_t b0 = rblk.b0;
    uint64_t b1 = rblk.b1;

    size_t r = pos % 64;
    uint64_t mask = (uint64_t(1) << r) - 1;

    /*
     * FIXME: gcc seems to generates a conditional jump, so
     * the code below needs to be replaced with __asm__().
     */
    uint64_t m = (pos & 64)? uint64_t(-1) : 0;
    uint64_t m0 = mask | m;
    uint64_t m1 = mask & m;

    ret += popcount64(b0 & m0);
    ret += popcount64(b1 & m1);

    return ret;
  }

  uint64_t  size_;
  std::vector<rBlock> rblk_;
}; /* SuccinctRank */

class SuccinctSelect {
 public:
  SuccinctSelect() : bit_(1), size_(0) {};
  explicit SuccinctSelect(const BitVector& bv,
                          RankPtr& rk, uint8_t bit) :
      bit_(bit), size_(0), rk_(rk) {init(bv);};
  ~SuccinctSelect() throw() {};

  uint64_t select(uint64_t pos) const {
    __assert(pos < size_);

    uint64_t rpos = rkQ_->rank(pos, 1) - 1;
    rBlock& rblk = rk_->get_rblock(rpos);

    uint64_t rem = pos - cumltv(rblk.rk, rpos * PRESUM_SZ);

    uint64_t rb = 0;
    block_t blk = 0;

    if (cumltv(rblk.b0sum, BSIZE) > rem) {
      blk = block(rblk.b0);
      rb = 0;
    } else {
      blk = block(rblk.b1);
      rb = BSIZE;
      rem -= cumltv(rblk.b0sum, BSIZE);
    }

    return rpos * PRESUM_SZ + rb + selectPos(blk, rem);
  }

 private:
  /*--- Private functions below ---*/
  void init(const BitVector& bv) {
    block_t   blk;
    BitVector Q;

    uint64_t sz = bv.length();
    size_t bsize = (sz + BSIZE - 1) / BSIZE;

    /* Calculate sz in advance */
    for (size_t i = 0; i < bsize; i++) {
      blk = block(bv.get_block(i));

      if ((i + 1) * BSIZE > sz) {
        uint64_t rem = sz - i * BSIZE;
        blk &= ((uint64_t(1) << rem) - 1);
      }

      size_ += popcount64(blk);
    }

    Q.init(size_);

    uint64_t qcount = 0;
    for (size_t i = 0; i < bsize; i++) {
      if (i % (PRESUM_SZ / BSIZE) == 0)
        Q.set_bit(qcount, 1);

      blk = block(bv.get_block(i));
      qcount += popcount64(blk);
    }

    RankPtr rkQ(new SuccinctRank(Q));
    rkQ_ = rkQ;
  }

  inline uint64_t cumltv(uint64_t val,
                         uint64_t pos) const {
    return (bit_)? val : pos - val;
  }

  inline block_t block(block_t b) const {
    return (bit_)? b : ~b;
  }

  uint8_t   bit_;
  uint64_t  size_;
  RankPtr   rkQ_;

  /*
  * A reference to the rank dictionary
  * of the orignal bit-vector.
  */
  RankPtr   rk_;
}; /* SuccinctSelect */

/* } namespace: */

class SuccinctBitVector {
 public:
  SuccinctBitVector() : rk_((SuccinctRank *)0),
    st0_((SuccinctSelect *)0), st1_((SuccinctSelect *)0) {};
  ~SuccinctBitVector() throw() {};

  /* Functions to initialize */
  void init(uint64_t size) {bv_.init(size);}

  void build() {
    if (bv_.length() == 0)
      throw "Not initialized yet: bv_";

    RankPtr rk(new SuccinctRank(bv_));
    SelectPtr st0(new SuccinctSelect(bv_, rk, 0));
    SelectPtr st1(new SuccinctSelect(bv_, rk, 1));

    rk_ = rk, st0_ = st0, st1_ = st1;
  }

  void set_bit(uint64_t pos, uint8_t bit) {
    if (pos >= bv_.length())
      throw "Invalid input: pos";
    if (bit > 1)
      throw "Invalid input: bit";

    bv_.set_bit(pos, bit);
  }

  bool lookup(uint64_t pos) const {
    if (pos >= bv_.length())
      throw "Invalid input: pos";

    return bv_.lookup(pos);
  }

  /* Rank & Select operations */
  uint64_t rank(uint64_t pos, uint8_t bit) const {
    if (pos >= bv_.length())
      throw "Invalid input: pos";
    if (bit > 1)
      throw "Invalid input: bit";

    return rk_->rank(pos, bit);
  }

  uint64_t select(uint64_t pos, uint8_t bit) const {
    if (pos >= bv_.get_none())
      throw "Invalid input: pos";
    if (bit > 1)
      throw "Invalid input: bit";

    return (bit)? st1_->select(pos) : st0_->select(pos);
  }

 private:
  /* A sequence of bit-array */
  BitVector bv_;

  /* A rank/select dictionary for dense */
  RankPtr   rk_;
  SelectPtr st0_;
  SelectPtr st1_;
}; /* SuccinctBitVector */

} /* dense */
} /* succinct */

#endif /* __SUCCINCTBITVECTOR_HPP__ */
