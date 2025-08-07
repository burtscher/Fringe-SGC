/*
Fringe-SGC is a C++/CUDA code for counting the number of occurrences of a subgraph in a larger graphs. 

Copyright (c) 2025, Cameron Bradley and Martin Burtscher

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at https://github.com/burtscher/Fringe-SGC.git.

Publication: This work is described in detail in the following paper.
Cameron Bradley and Martin Burtscher. "Fringe-SGC: Counting Subgraphs with Fringe Vertices" Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC’25). November 2025.


Sponsor: This work has been supported by the National Science Foundation under Award #1955367.
*/
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
#include <cuda.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include "ECLgraph.h"
#include "patterns.h"
#include "fringes.h"


/*
 The input graph must meet the following conditions, which are not checked:
 - The graph cannot contain self-edges
 - The graph cannot contain multiple edges between the same pair of vertices in the same direction
 - The graph must be undirected, i.e., every edge must appear twice in the CSR format, once in each direction
 - The individual adjacency lists must be sorted in increasing order
*/


using ull = unsigned long long;

static const int constexpr iu = 1 << 0;
static const int constexpr iv = 1 << 1;
static const int constexpr iw = 1 << 2;
static const int constexpr iuv = iu | iv;
static const int constexpr iuw = iu | iw;
static const int constexpr ivw = iv | iw;
static const int constexpr iuvw = iu | iv | iw;


static const int constexpr TPB = 512;  // threads per block
static const int constexpr WS = 32;  // warp size

static __device__ ull total = 0;
static __device__ int wlsize = 0;
static __device__ int wlpos = 0;
static __device__ int maxdeg = 0;
static __device__ int d_fr [6][8];


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
  #define __kernel__ __global__ __launch_bounds__(TPB, 2048 / TPB)
#else
  #define __kernel__ __global__ __launch_bounds__(TPB, 1536 / TPB)
#endif


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
static inline __device__ int __reduce_add_sync(const int mask, int val)
{
  val += __shfl_xor_sync(~0, val, 1);
  val += __shfl_xor_sync(~0, val, 2);
  val += __shfl_xor_sync(~0, val, 4);
  val += __shfl_xor_sync(~0, val, 8);
  val += __shfl_xor_sync(~0, val, 16);
  return val;
}
#endif


static inline void CheckCuda(const int line)
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d on line %d: %s\n\n", e, line, cudaGetErrorString(e));
    exit(-1);
  }
}


struct CPUTimer
{
  timeval beg, end;
  CPUTimer() {}
  ~CPUTimer() {}
  void start() {gettimeofday(&beg, NULL);}
  double elapsed() {gettimeofday(&end, NULL); return end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;}
};


static inline ECLgraph matchingOrder2ECLgraph(MatchingOrder mo)
{
  ECLgraph g;

  int n = mo.nodes;
  for (int i = 0; i < mo.anchorsets; i++) {
    n += mo.fcnt[i];
  }
  g.nodes = n;
  g.nindex = new int [n + 1];
  g.nindex[0] = 0;
  int p = 1;

  int m = mo.startdeg;
  g.nindex[p++] = m;
  for (int i = 0; i < mo.nodes - 1; i++) {
    m += mo.destdeg[i];
    g.nindex[p++] = m;
  }
  for (int i = 0; i < mo.anchorsets; i++) {
    const int deg = __builtin_popcount(mo.anch[i]);
    for (int j = 0; j < mo.fcnt[i]; j++) {
      m += deg;
      g.nindex[p++] = m;
    }
  }
  g.edges = m;
  g.nlist = new int [m];

  int cnt [MaxPatternSize];
  for (int i = 0; i < n; i++) {
    cnt[i] = g.nindex[i];
  }

  for (int i = 0; i < mo.nodes - 1; i++) {
    for (int j = 0; j <= i; j++) {
      if (mo.list[i] & (1 << j)) {
        g.nlist[cnt[i + 1]++] = j;
        g.nlist[cnt[j]++] = i + 1;
      }
    }
  }
  int v = mo.nodes;
  for (int i = 0; i < mo.anchorsets; i++) {
    int bmp = mo.anch[i];
    for (int k = 0; k < mo.fcnt[i]; k++) {
      for (int j = 0; j < MaxCoreSize; j++) {
        if (bmp & (1 << j)) {
          g.nlist[cnt[j]++] = v;
          g.nlist[cnt[v]++] = j;
        }
      }
      v++;
    }
  }

  for (int i = 0; i < n; i++) {
    //assert(cnt[i] == g.nindex[i + 1]);
    if (g.nindex[i + 1] - g.nindex[i] > 1) {
      std::sort(&g.nlist[g.nindex[i]], &g.nlist[g.nindex[i + 1]]);
    }
  }

  g.eweight = NULL;
  return g;
}


static __device__ __host__ inline ull nCk(const int n, int k)
{
  k = min(k, n - k);
  if (k < 0) return 0;
  if (k == 0) return 1;
  ull r = n;
  for (int i = 2; i <= k; i++) r = r * (n + 1 - i) / i;
  return r;
}


static inline __device__ __host__ bool contains(const int v, int l, int r, const int* const __restrict__ nlist)
{
  int m = (l + r) / 2;
  while (m != l) {
    if (v < nlist[m]) {
      r = m;
    } else {
      l = m;
    }
    m = (l + r) / 2;
  }
  return (v == nlist[m]);
}


static inline __device__ int common2(const int beg1, const int end1, const int beg2, const int end2, const int* const __restrict__ nlist)  // warp-based
{
  const int deg1 = end1 - beg1;
  const int deg2 = end2 - beg2;
  const int lane = threadIdx.x % WS;
  const int beg = (deg1 < deg2) ? beg1 : beg2;
  const int end = (deg1 < deg2) ? end1 : end2;
  const int left = (deg1 < deg2) ? beg2 : beg1;
  const int right = (deg1 < deg2) ? end2 : end1;
  int c12 = 0;
  for (int i = beg + lane; i < end; i += WS) {
    const int w = nlist[i];
    if (contains(w, left, right, nlist)) c12++;
  }
  return __reduce_add_sync(~0, c12);
}


static inline __device__ ull fc3(const int fr [8], const int venn [8])
{
  const int fr_uvw = fr[iuvw];
  const int fr_uv = fr[iuv];
  const int fr_uw = fr[iuw];
  const int fr_vw = fr[ivw];
  const int fr_u = fr[iu];
  const int fr_v = fr[iv];
  const int fr_w = fr[iw];
  int vd_uvw = venn[iuvw];
  int vd_uv = venn[iuv];
  int vd_uw = venn[iuw];
  int vd_vw = venn[ivw];
  int vd_u = venn[iu];
  int vd_v = venn[iv];
  int vd_w = venn[iw];
  ull count = 0;

  // uvw
  if (vd_uvw < fr_uvw) return 0;
  const ull cuvw = nCk(vd_uvw, fr_uvw);
  vd_uvw -= fr_uvw;

  // uv
  for (int i = min(vd_uvw, fr_uv); i >= 0; i--) {
    if (vd_uv < fr_uv - i) break;
    const ull cuvw_uv = cuvw * nCk(vd_uvw, i) * nCk(vd_uv, fr_uv - i);
    vd_uvw -= i;
    vd_uv -= fr_uv - i;

    // uw
    for (int j = min(vd_uvw, fr_uw); j >= 0; j--) {
      if (vd_uw < fr_uw - j) break;
      const ull cuvw_uv_uw = cuvw_uv * nCk(vd_uvw, j) * nCk(vd_uw, fr_uw - j);
      vd_uvw -= j;
      vd_uw -= fr_uw - j;

      // vw
      for (int k = min(vd_uvw, fr_vw); k >= 0; k--) {
        if (vd_vw < fr_vw - k) break;
        const ull cuvw_uv_uw_vw = cuvw_uv_uw * nCk(vd_uvw, k) * nCk(vd_vw, fr_vw - k);
        vd_uvw -= k;
        vd_vw -= fr_vw - k;

        // u
        for (int a = min(vd_uvw, fr_u); a >= 0; a--) {
          const ull tmpa = cuvw_uv_uw_vw * nCk(vd_uvw, a);
          vd_uvw -= a;
          int vd_uvw_vw = vd_uvw + vd_vw;
          for (int b = min(vd_uv, fr_u - a); b >= 0; b--) {
            const ull tmpb = tmpa * nCk(vd_uv, b);
            vd_uv -= b;
            for (int c = min(vd_uw, fr_u - a - b); c >= 0; c--) {
              if (vd_u < fr_u - a - b - c) break;
              const ull cuvw_uv_uw_vw_u = tmpb * nCk(vd_uw, c) * nCk(vd_u, fr_u - a - b - c);
              vd_uw -= c;

              // v
              for (int d = min(vd_uvw_vw, fr_v); d >= 0; d--) {
                if (vd_uv + vd_v < fr_v - d) break;
                const ull cuvw_uv_uw_vw_u_v = cuvw_uv_uw_vw_u * nCk(vd_uvw_vw, d) * nCk(vd_uv + vd_v, fr_v - d);
                vd_uvw_vw -= d;

                // w
                if (vd_uvw_vw + vd_uw + vd_w >= fr_w) {
                  const ull cuvw_uv_uw_vw_u_v_w = cuvw_uv_uw_vw_u_v * nCk(vd_uvw_vw + vd_uw + vd_w, fr_w);
                  count += cuvw_uv_uw_vw_u_v_w;
                }

                vd_uvw_vw += d;
              }

              vd_uw += c;
            }
            vd_uv += b;
          }
          vd_uvw += a;
        }

        vd_uvw += k;
        vd_vw += fr_vw - k;
      }

      vd_uvw += j;
      vd_uw += fr_uw - j;
    }

    vd_uvw += i;
    vd_uv += fr_uv - i;
  }

//  vd_uvw += fr_uvw;

  return count;
}


/**************************************************************/
/* General core code                                          */
/**************************************************************/


/*
large cores with just one fringe
 only slow for patterns close to cliques
 - combine Venn sets for later iterations in functionX
 - add symmetry breaking in general code
 - exploit rotations
 - exploit hub vertices for cliques etc.
*/


static inline __device__ void commonX(const ECLgraph g, const int corenodes, int* const __restrict__ venn, const int* const __restrict__ stack, const byte nonanchors)  // warp-based
{
  const int lane = threadIdx.x % WS;
  for (int i = lane + 1; i < (1 << corenodes); i += WS) {
    venn[i] = 0;
  }
  __syncwarp();

  for (int i = corenodes - 1; i >= 0; i--) {
    if ((1 << i) & nonanchors) continue;
    const int w_n = stack[i];
    const int w_n_beg = g.nindex[w_n];
    const int w_n_end = g.nindex[w_n + 1];
    for (int j = w_n_beg + lane; j < w_n_end; j += WS) {
      const int x = g.nlist[j];
      int bmp = 1 << i;
      for (int k = 0; k < i; k++) {
        if ((1 << k) & nonanchors) continue;
        const int w_t = stack[k];
        if (contains(x, g.nindex[w_t], g.nindex[w_t + 1], g.nlist)) {
          bmp |= 1 << k;
        }
      }
      atomicAdd_block(&venn[bmp], 1);
    }
  }
  __syncwarp();

  // correct venn entries
  static_assert(MaxCoreSize <= 6);  // no more than 6 bits supported
  const int top = 1 << corenodes;
  for (int i = corenodes - 2; i >= 0; i--) {
    const int j = (1 << i) + lane;
    const int hi = (2 << i);
    if (j < hi) {
      int sum = 0;
      int pos = j + hi;
      while (pos < top) {
        sum += venn[pos];
        pos += hi;
      }
      venn[j] -= sum;
    }
    __syncwarp();
  }

  // do not count if vertex is on stack
  if (lane < corenodes) {
    const int x = stack[lane];
    int bmp = 0;
    for (int k = 0; k < corenodes; k++) {
      if ((1 << k) & nonanchors) continue;
      const int w_t = stack[k];
      if (contains(x, g.nindex[w_t], g.nindex[w_t + 1], g.nlist)) {
        bmp |= 1 << k;
      }
    }
    atomicSub_block(&venn[bmp], 1);
  }
  __syncwarp();
}


static __device__ ull functionX(const MatchingOrder mo, int* const __restrict__ venn, char* const __restrict__ fr, const byte nonanchors)  // thread-based
{
  const int constexpr size = 665;  // up to 3^(#bits) - 2^(#bits) levels needed
  static_assert(MaxCoreSize <= 6);

  int savea [size];
  ull saveprod [size];
  int saveidx [size];
  int savevenn [size];
  char savefr [size];

  const int top = ((1 << mo.nodes) - 1) ^ nonanchors;
  int asi = mo.anchorsets - 1;  // anchor set index
  ull sum = 0;
  ull prod = 1;

  // prep first level
  int level = 0;
  int anch = mo.anch[asi];
  int idx = anch;
  int a = min(venn[idx], fr[anch]);

  // iterate
  do {
    savea[level] = a;
    saveprod[level] = prod;
    saveidx[level] = idx;
    savevenn[level] = venn[idx];
    savefr[level] = fr[anch];
    if (a < 0) {
      prod = 0;
    } else {
      if (idx == top) {
        prod *= nCk(venn[idx], fr[anch]);
        venn[idx] -= fr[anch];
        fr[anch] = 0;
        if (asi == 0) {
          // we are done
          sum += prod;
          prod = 0;
        }
      } else {
        prod *= nCk(venn[idx], a);
        venn[idx] -= a;
        fr[anch] -= a;
      }
    }
    if (prod == 0) {
      // prior level
      level--;
      if (level < 0) break;

      a = savea[level] - 1;
      prod = saveprod[level];
      idx = saveidx[level];
      if (idx == top) {
        asi++;
        anch = mo.anch[asi];
      }
      venn[idx] = savevenn[level];
      fr[anch] = savefr[level];
    } else {
      // next level
      level++;
      //assert(level < size);
      if (idx == top) {
        asi--;
        anch = mo.anch[asi];
        idx = anch;
      } else {
        idx = (((idx | nonanchors) + 1) | anch) & ~nonanchors;
      }
      a = (idx == top) ? 0 : min(venn[idx], fr[anch]);
    }
  } while (true);
  fr[mo.anch[mo.anchorsets - 1]] = savefr[0];

  return sum;
}


template <int level>
static inline __device__ ull general_core_level(const MatchingOrder mo, const ECLgraph g, int* const __restrict__ stack, int* const __restrict__ venn, char* const __restrict__ fr, const byte nonanchors)
{
  const int idx = threadIdx.x + blockIdx.x * TPB;
  const int lane = threadIdx.x % WS;
  const int w_n = stack[__ffs(mo.list[level]) - 1];
  const int w_n_beg = g.nindex[w_n];
  const int w_n_end = g.nindex[w_n + 1];

  ull sum = 0;
  for (int j = w_n_beg + lane; __any_sync(~0, j < w_n_end); j += WS) {
    bool active = (j < w_n_end);
    int x;
    if (active) {
      x = g.nlist[j];
      const int x_beg = g.nindex[x];
      const int x_end = g.nindex[x + 1];
      const int x_deg = x_end - x_beg;
      active = (x_deg >= mo.destdeg[level]);  // deg(x) >= mindeg
      if (active) {
        const int list = mo.list[level];
        int bmp = list ^ ((2 << level) - 1);
        while (bmp != 0) {
          const int z = stack[__ffs(bmp) - 1];
          bmp &= bmp - 1;
          if (x == z) {  // skipping over ...
            active = false;
            break;
          }
        }
        if (active) {
          int bmp = list & (list - 1);  // skip itself
          while (bmp != 0) {
            const int z = stack[__builtin_ffs(bmp) - 1];
            bmp &= bmp - 1;
            if (!contains(x, g.nindex[z], g.nindex[z + 1], g.nlist)) {  // connecting to ...
              active = false;
              break;
            }
          }
        }
      }
    }

    int bal = __ballot_sync(~0, active);
    if ((level > 1) && (level == mo.nodes - 2)) {
      while (bal != 0) {
        const int old = bal;
        bal &= bal - 1;
        const int who = __ffs(bal ^ old) - 1;
        if (who == lane) stack[level + 1] = x;
        __syncwarp();
        commonX(g, mo.nodes, &venn[(idx - lane + who) << MaxCoreSize], stack, nonanchors);
      }
      if (active) {
        sum += functionX(mo, &venn[idx << MaxCoreSize], fr, nonanchors);
      }
    } else {
      while (bal != 0) {
        const int old = bal;
        bal &= bal - 1;
        const int who = __ffs(bal ^ old) - 1;
        if (who == lane) stack[level + 1] = x;
        __syncwarp();
        if constexpr (level < 4) {
          sum += general_core_level<level + 1>(mo, g, stack, venn, fr, nonanchors);
        } else {
          printf("ERROR: too many levels\n");
          __trap();
        }
      }
    }
  }

  return sum;
}


static __kernel__ void general_core_level0(const MatchingOrder mo, const ECLgraph g, const int2* const __restrict__ wl, const byte nonanchors, int* const __restrict__ venn, char* const __restrict__ gfr)
{
  const int constexpr warps = TPB / WS;
  __shared__ int st [warps][MaxCoreSize];

  const int idx = threadIdx.x + blockIdx.x * TPB;
  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;
  int* const __restrict__ stack = st[warp];
  char* const __restrict__ fr = &gfr[idx << MaxCoreSize];

  // transform
  for (int i = 1; i < (1 << mo.nodes); i++) {
    fr[i] = 0;
  }
  for (int i = 0; i < mo.anchorsets; i++) {
    const int as = mo.anch[i];
    fr[as] = mo.fcnt[i];
  }

  ull sum = 0;
  do {
    const int i = atomicAdd(&wlpos, 1);
    bool active = (i < wlsize);
    if (__all_sync(~0, !active)) break;

    int u, v;
    if (active) {
      const int2 ele = wl[i];  // u, v
      u = ele.x;
      v = ele.y;
    }

    int bal = __ballot_sync(~0, active);
    while (bal != 0) {
      const int old = bal;
      bal &= bal - 1;
      const int who = __ffs(bal ^ old) - 1;
      if (lane == who) {
        stack[0] = u;
        stack[1] = v;
      }
      sum += general_core_level<1>(mo, g, stack, venn, fr, nonanchors);
    }
  } while (true);

  atomicAdd(&total, sum);
}


static __kernel__ void init_wl_general(const ECLgraph g, int2* const __restrict__ wl, const int lim1, const int lim2)
{
  const int u = threadIdx.x + blockIdx.x * TPB;
  int u_beg, u_end;
  bool active = false;
  if (u < g.nodes) {
    u_beg = g.nindex[u];
    u_end = g.nindex[u + 1];
    const int u_deg = u_end - u_beg;
    if (u_deg >= lim1) {
      active = true;
      if (u_deg < WS) {
        active = false;
        for (int i = u_beg; i < u_end; i++) {
          const int v = g.nlist[i];
          const int v_beg = g.nindex[v];
          const int v_end = g.nindex[v + 1];
          const int v_deg = v_end - v_beg;
          if (v_deg >= lim2) {
            const int loc = atomicAdd(&wlsize, 1);
            wl[loc] = int2{u, v};
          }
        }
      }
    }
  }
  int bal = __ballot_sync(~0, active);
  const int lane = threadIdx.x % WS;
  while (bal != 0) {
    const int old = bal;
    bal &= bal - 1;
    const int who = __ffs(bal ^ old) - 1;
    const int w_u = __shfl_sync(~0, u, who);
    const int w_u_beg = __shfl_sync(~0, u_beg, who);
    const int w_u_end = __shfl_sync(~0, u_end, who);
    for (int i = w_u_beg + lane; i < w_u_end; i += WS) {
      const int v = g.nlist[i];
      const int v_beg = g.nindex[v];
      const int v_end = g.nindex[v + 1];
      const int v_deg = v_end - v_beg;
      if (v_deg >= lim2) {
        const int loc = atomicAdd(&wlsize, 1);
        wl[loc] = int2{w_u, v};
      }
    }
  }
}


static ull generalCore(const MatchingOrder mo, const ECLgraph g, int2* const wl, const int SMs, const int mTpSM)
{
  init_wl_general<<<(g.nodes + TPB - 1) / TPB, TPB>>>(g, wl, mo.startdeg, mo.destdeg[0]);

  // core vertices that are anchors
  byte anchors = 0;
  for (int i = 0; i < mo.anchorsets; i++) {
    anchors |= mo.anch[i];
  }
  const byte nonanchors = anchors ^ ((1 << mo.nodes) - 1);

  const int blocks = SMs * (mTpSM / TPB);
  int* d_venn;
  char* d_fr;
  cudaMalloc((void**)&d_venn, sizeof(int) * blocks * TPB * (1 << MaxCoreSize));
  cudaMalloc((void**)&d_fr, sizeof(char) * blocks * TPB * (1 << MaxCoreSize));
  //CheckCuda(__LINE__);
  general_core_level0<<<blocks, TPB>>>(mo, g, wl, nonanchors, d_venn, d_fr);
  cudaFree(d_venn);
  cudaFree(d_fr);
  //CheckCuda(__LINE__);

  ull count;
  cudaMemcpyFromSymbol(&count, total, sizeof(total));

  return count;
}


/**************************************************************/
/* 3-vertex triangle-core code                                */
/**************************************************************/


static inline __device__ int com3_clique_triangle(const int beg1, const int end1, const int beg2, const int end2, const int beg3, const int end3, const int* const __restrict__ nlist, const int top)
{
  const int lane = threadIdx.x % WS;
  int sum = 0;
  for (int j = beg1 + lane; j < end1; j += WS) {
    const int q = nlist[j];
    if (q >= top) break;
    if (contains(q, beg2, end2, nlist)) {
      if (contains(q, beg3, end3, nlist)) {
        sum++;
      }
    }
  }
  return sum;
}


static inline __device__ int common3_clique_triangle(const int beg1, const int end1, const int beg2, const int end2, const int* const __restrict__ nindex, const int* const __restrict__ nlist, const int v)
{
  const int deg1 = end1 - beg1;
  const int deg2 = end2 - beg2;
  const int lane = threadIdx.x % WS;
  const int beg = (deg1 < deg2) ? beg1 : beg2;
  const int end = (deg1 < deg2) ? end1 : end2;
  const int left = (deg1 < deg2) ? beg2 : beg1;
  const int right = (deg1 < deg2) ? end2 : end1;
  int sum = 0;
  for (int i = beg + lane; __any_sync(~0, i < end); i += WS) {
    const int w = (i < end) ? nlist[i] : INT_MAX;
    if (__all_sync(~0, w >= v)) break;
    int beg3, end3;
    bool found = false;
    if (w < v) {
      if (contains(w, left, right, nlist)) {
        beg3 = nindex[w];
        end3 = nindex[w + 1];
        const int deg3 = end3 - beg3;
        found = (deg3 >= 3);
      }
    }
    int bal = __ballot_sync(~0, found);
    while (bal != 0) {
      const int old = bal;
      bal &= bal - 1;
      const int who = __ffs(bal ^ old) - 1;
      const int w_w = __shfl_sync(~0, w, who);
      const int w_w_beg = __shfl_sync(~0, beg3, who);
      const int w_w_end = __shfl_sync(~0, end3, who);
      const int deg3 = w_w_end - w_w_beg;
      if ((deg1 <= deg2) && (deg1 <= deg3)) {
        if (deg2 <= deg3) {
          sum += com3_clique_triangle(beg1, end1, beg2, end2, w_w_beg, w_w_end, nlist, w_w);
        } else {
          sum += com3_clique_triangle(beg1, end1, w_w_beg, w_w_end, beg2, end2, nlist, w_w);
        }
      } else if ((deg2 <= deg1) && (deg2 <= deg3)) {
        if (deg1 <= deg3) {
          sum += com3_clique_triangle(beg2, end2, beg1, end1, w_w_beg, w_w_end, nlist, w_w);
        } else {
          sum += com3_clique_triangle(beg2, end2, w_w_beg, w_w_end, beg1, end1, nlist, w_w);
        }
      } else {
        if (deg1 <= deg2) {
          sum += com3_clique_triangle(w_w_beg, w_w_end, beg1, end1, beg2, end2, nlist, w_w);
        } else {
          sum += com3_clique_triangle(w_w_beg, w_w_end, beg2, end2, beg1, end1, nlist, w_w);
        }
      }
    }
  }
  return sum;
}


static __kernel__ void clique(const ECLgraph g, const int2* const __restrict__ wl)
{
  ull sum = 0;

  do {
    const int i = atomicAdd(&wlpos, 1);
    if (__all_sync(~0, i >= wlsize)) break;

    bool active = (i < wlsize);
    int v, u_beg, u_end, v_beg, v_end;
    if (active) {
      const int2 ele = wl[i];  // u, v
      const int u = ele.x;
      v = ele.y;
      u_beg = g.nindex[u];
      u_end = g.nindex[u + 1];
      v_beg = g.nindex[v];
      v_end = g.nindex[v + 1];
    }

    int bal = __ballot_sync(~0, active);
    while (bal != 0) {
      const int old = bal;
      bal &= bal - 1;
      const int who = __ffs(bal ^ old) - 1;
      const int w_v = __shfl_sync(~0, v, who);
      const int w_u_beg = __shfl_sync(~0, u_beg, who);
      const int w_u_end = __shfl_sync(~0, u_end, who);
      const int w_v_beg = __shfl_sync(~0, v_beg, who);
      const int w_v_end = __shfl_sync(~0, v_end, who);
      sum += common3_clique_triangle(w_u_beg, w_u_end, w_v_beg, w_v_end, g.nindex, g.nlist, w_v);
    }
  } while (true);

  atomicAdd(&total, sum);
}


static inline __device__ ull com3_tailed_clique_triangle(const int beg1, const int end1, const int beg2, const int end2, const int beg3, const int end3, const int* const __restrict__ nindex, const int* const __restrict__ nlist, const int top, const int tails, const ull c123)
{
  const int lane = threadIdx.x % WS;
  ull sum = 0;
  for (int j = beg1 + lane; j < end1; j += WS) {
    const int q = nlist[j];
    if (q >= top) break;
    if (contains(q, beg2, end2, nlist)) {
      if (contains(q, beg3, end3, nlist)) {
        const int deg4 = nindex[q + 1] - nindex[q];
        sum += c123 + nCk(deg4 - 3, tails);
      }
    }
  }
  return sum;
}


static inline __device__ ull common3_tailed_clique_triangle(const int beg1, const int end1, const int beg2, const int end2, const int* const __restrict__ nindex, const int* const __restrict__ nlist, const int v, const int tails, const ull c12)
{
  const int deg1 = end1 - beg1;
  const int deg2 = end2 - beg2;
  const int lane = threadIdx.x % WS;
  const int beg = (deg1 < deg2) ? beg1 : beg2;
  const int end = (deg1 < deg2) ? end1 : end2;
  const int left = (deg1 < deg2) ? beg2 : beg1;
  const int right = (deg1 < deg2) ? end2 : end1;
  ull c123, sum = 0;
  for (int i = beg + lane; __any_sync(~0, i < end); i += WS) {
    const int w = (i < end) ? nlist[i] : INT_MAX;
    if (__all_sync(~0, w >= v)) break;
    int beg3, end3;
    bool found = false;
    if (w < v) {
      if (contains(w, left, right, nlist)) {
        beg3 = nindex[w];
        end3 = nindex[w + 1];
        const int deg3 = end3 - beg3;
        if (deg3 >= 3) {
          c123 = c12 + nCk(deg3 - 3, tails);
          found = true;
        }
      }
    }
    int bal = __ballot_sync(~0, found);
    while (bal != 0) {
      const int old = bal;
      bal &= bal - 1;
      const int who = __ffs(bal ^ old) - 1;
      const int w_w = __shfl_sync(~0, w, who);
      const int w_w_beg = __shfl_sync(~0, beg3, who);
      const int w_w_end = __shfl_sync(~0, end3, who);
      const ull w_c123 = __shfl_sync(~0, c123, who);
      const int deg3 = w_w_end - w_w_beg;
      if ((deg1 <= deg2) && (deg1 <= deg3)) {
        if (deg2 <= deg3) {
          sum += com3_tailed_clique_triangle(beg1, end1, beg2, end2, w_w_beg, w_w_end, nindex, nlist, w_w, tails, w_c123);
        } else {
          sum += com3_tailed_clique_triangle(beg1, end1, w_w_beg, w_w_end, beg2, end2, nindex, nlist, w_w, tails, w_c123);
        }
      } else if ((deg2 <= deg1) && (deg2 <= deg3)) {
        if (deg1 <= deg3) {
          sum += com3_tailed_clique_triangle(beg2, end2, beg1, end1, w_w_beg, w_w_end, nindex, nlist, w_w, tails, w_c123);
        } else {
          sum += com3_tailed_clique_triangle(beg2, end2, w_w_beg, w_w_end, beg1, end1, nindex, nlist, w_w, tails, w_c123);
        }
      } else {
        if (deg1 <= deg2) {
          sum += com3_tailed_clique_triangle(w_w_beg, w_w_end, beg1, end1, beg2, end2, nindex, nlist, w_w, tails, w_c123);
        } else {
          sum += com3_tailed_clique_triangle(w_w_beg, w_w_end, beg2, end2, beg1, end1, nindex, nlist, w_w, tails, w_c123);
        }
      }
    }
  }
  return sum;
}


static __kernel__ void tailed_clique_triangle(const ECLgraph g, const int2* const __restrict__ wl, const int tails)
{
  ull sum = 0;

  do {
    const int i = atomicAdd(&wlpos, 1);
    if (__all_sync(~0, i >= wlsize)) break;

    bool active = (i < wlsize);
    int v, u_beg, u_end, v_beg, v_end;
    ull c12;
    if (active) {
      const int2 ele = wl[i];  // u, v
      const int u = ele.x;
      v = ele.y;
      u_beg = g.nindex[u];
      u_end = g.nindex[u + 1];
      v_beg = g.nindex[v];
      v_end = g.nindex[v + 1];
      const ull c1 = nCk(u_end - u_beg - 3, tails);
      const ull c2 = nCk(v_end - v_beg - 3, tails);
      c12 = c1 + c2;
    }

    int bal = __ballot_sync(~0, active);
    while (bal != 0) {
      const int old = bal;
      bal &= bal - 1;
      const int who = __ffs(bal ^ old) - 1;
      const int w_v = __shfl_sync(~0, v, who);
      const int w_u_beg = __shfl_sync(~0, u_beg, who);
      const int w_u_end = __shfl_sync(~0, u_end, who);
      const int w_v_beg = __shfl_sync(~0, v_beg, who);
      const int w_v_end = __shfl_sync(~0, v_end, who);
      const ull w_c12 = __shfl_sync(~0, c12, who);
      sum += common3_tailed_clique_triangle(w_u_beg, w_u_end, w_v_beg, w_v_end, g.nindex, g.nlist, w_v, tails, w_c12);
    }
  } while (true);

  atomicAdd(&total, sum);
}


static inline __device__ void com3(const int beg1, const int end1, const int beg2, const int end2, const int beg3, const int end3, const int* const __restrict__ nlist, const int who, int venn [8], const int p12)
{
  const int lane = threadIdx.x % WS;
  int t13 = 0;
  int t23 = 0;
  int t123 = 0;
  for (int j = beg3 + lane; j < end3; j += WS) {
    const int q = nlist[j];
    const bool b13 = contains(q, beg1, end1, nlist);
    const bool b23 = contains(q, beg2, end2, nlist);
    t13 += b13;
    t23 += b23;
    t123 += b13 & b23;
  }
  const int c123 = __reduce_add_sync(~0, t123);
  const int c12 = p12 - (c123 + 1);
  const int c13 = __reduce_add_sync(~0, t13) - (c123 + 1);
  const int c23 = __reduce_add_sync(~0, t23) - (c123 + 1);
  const int c1 = end1 - beg1 - (c12 + c13 + (c123 + 2));
  const int c2 = end2 - beg2 - (c12 + c23 + (c123 + 2));
  const int c3 = end3 - beg3 - (c13 + c23 + (c123 + 2));
  if (lane == who) {
    venn[iu] = c1;
    venn[iv] = c2;
    venn[iw] = c3;
    venn[iuv] = c12;
    venn[iuw] = c13;
    venn[ivw] = c23;
    venn[iuvw] = c123;
  }
}


static inline __device__ ull common3_triangle(const int beg1, const int end1, const int beg2, const int end2, const int* const __restrict__ nindex, const int* const __restrict__ nlist, const int v, const int sum1, const int sum2, const int sum3, const int rotations)
{
  __shared__ int venn [TPB * 8];

  const int deg1 = end1 - beg1;
  const int deg2 = end2 - beg2;
  const int lane = threadIdx.x % WS;
  const int warpoffs = (threadIdx.x / WS) * WS;
  const int beg = (deg1 < deg2) ? beg1 : beg2;
  const int end = (deg1 < deg2) ? end1 : end2;
  const int left = (deg1 < deg2) ? beg2 : beg1;
  const int right = (deg1 < deg2) ? end2 : end1;
  const int thres2 = max(max(sum1, sum3 - deg1 - deg2), max(sum2 - deg1, sum2 - deg2));

  int t12 = 0;
  for (int j = beg + lane; j < end; j += WS) {
    const int q = nlist[j];
    if (contains(q, left, right, nlist)) t12++;
  }
  const int p12 = __reduce_add_sync(~0, t12);  // not corrected

  ull sum = 0;
  for (int i = beg + lane; __any_sync(~0, i < end); i += WS) {
    const int w = (i < end) ? nlist[i] : INT_MAX;
    if (__all_sync(~0, w >= v)) break;
    int beg3, end3;
    bool found = false;
    if (w < v) {
      if (contains(w, left, right, nlist)) {
        beg3 = nindex[w];
        end3 = nindex[w + 1];
        const int deg3 = end3 - beg3;
        found = (deg3 >= thres2);
      }
    }
    int bal = __ballot_sync(~0, found);
    int cnt = 0;
    while (bal != 0) {
      const int old = bal;
      bal &= bal - 1;
      const int who = __ffs(bal ^ old) - 1;
      const int w_w_beg = __shfl_sync(~0, beg3, who);
      const int w_w_end = __shfl_sync(~0, end3, who);
      com3(beg1, end1, beg2, end2, w_w_beg, w_w_end, nlist, cnt, &venn[(warpoffs + cnt) * 8], p12);
      cnt++;
    }

    __syncwarp();
    const int tot = rotations * cnt;
    for (int i = lane; i < tot; i += WS) {
      const int r = i / cnt;
      const int p = i % cnt;
      sum += fc3(d_fr[r], &venn[(warpoffs + p) * 8]);
    }
  }

  return sum;
}


static __kernel__ void general_triangle_core(const ECLgraph g, const int2* const __restrict__ wl, const int rotations, const int sum1, const int sum2, const int sum3)
{
  ull sum = 0;

  do {
    const int i = atomicAdd(&wlpos, 1);
    if (__all_sync(~0, i >= wlsize)) break;

    bool active = (i < wlsize);
    int v, u_beg, u_end, v_beg, v_end;
    if (active) {
      const int2 ele = wl[i];  // u, v
      const int u = ele.x;
      v = ele.y;
      u_beg = g.nindex[u];
      u_end = g.nindex[u + 1];
      v_beg = g.nindex[v];
      v_end = g.nindex[v + 1];
    }

    int bal = __ballot_sync(~0, active);
    while (bal != 0) {
      const int old = bal;
      bal &= bal - 1;
      const int who = __ffs(bal ^ old) - 1;
      const int w_v = __shfl_sync(~0, v, who);
      const int w_u_beg = __shfl_sync(~0, u_beg, who);
      const int w_u_end = __shfl_sync(~0, u_end, who);
      const int w_v_beg = __shfl_sync(~0, v_beg, who);
      const int w_v_end = __shfl_sync(~0, v_end, who);
      sum += common3_triangle(w_u_beg, w_u_end, w_v_beg, w_v_end, g.nindex, g.nlist, w_v, sum1, sum2, sum3, rotations);
    }
  } while (true);

  atomicAdd(&total, sum);
}


static __kernel__ void init_sort(const ECLgraph g, ull* const __restrict__ sort, const int shift)
{
  const int u = threadIdx.x + blockIdx.x * TPB;
  if (u < g.nodes) {
    const int beg = g.nindex[u];
    const int end = g.nindex[u + 1];
    const int deg = end - beg;
    sort[u] = ((ull)deg << shift) + u;
    atomicMax(&maxdeg, deg);
  }
}


static __kernel__ void create_map(const int nodes, const ull* const __restrict__ sort, int* const __restrict__ map, const ull mask)
{
  const int u = threadIdx.x + blockIdx.x * TPB;
  if (u < nodes) {
    const ull val = sort[u];
    map[val & mask] = u;
  }
}


static __kernel__ void create_edge_list(const ECLgraph g, const int* const __restrict__ map, ull* const __restrict__ sort, const int shift)
{
  __shared__ ull s_mu [TPB];
  __shared__ int s_beg [TPB];
  __shared__ int s_end [TPB];
  __shared__ int s_num;
  if (threadIdx.x == 0) s_num = 0;
  __syncthreads();

  const int u = threadIdx.x + blockIdx.x * TPB;
  int beg, end, deg = 0;
  ull mu;
  if (u < g.nodes) {
    mu = ((ull)map[u]) << shift;
    beg = g.nindex[u];
    end = g.nindex[u + 1];
    deg = end - beg;
    if (deg < WS) {
      for (int i = beg; i < end; i++) {
        const int mn = map[g.nlist[i]];
        sort[i] = mu + mn;
      }
    }
  }
  int bal = __ballot_sync(~0, (deg >= WS) && (deg < TPB));
  const int lane = threadIdx.x % WS;
  while (bal != 0) {
    const int old = bal;
    bal &= bal - 1;
    const int who = __ffs(bal ^ old) - 1;
    const ull w_mu = __shfl_sync(~0, mu, who);
    const int w_beg = __shfl_sync(~0, beg, who);
    const int w_end = __shfl_sync(~0, end, who);
    for (int i = w_beg + lane; i < w_end; i += WS) {
      const int mn = map[g.nlist[i]];
      sort[i] = w_mu + mn;
    }
  }
  if (deg >= TPB) {
    const int pos = atomicAdd_block(&s_num, 1);
    s_mu[pos] = mu;
    s_beg[pos] = beg;
    s_end[pos] = end;
  }
  __syncthreads();

  const int num = s_num;
  for (int j = 0; j < num; j++) {
    const int b_beg = s_beg[j];
    const int b_end = s_end[j];
    const ull b_mu = s_mu[j];
    for (int i = b_beg + threadIdx.x; i < b_end; i += TPB) {
      const int mn = map[g.nlist[i]];
      sort[i] = b_mu + mn;
    }
  }
}


static __kernel__ void create_nindex(ECLgraph g, const ull* const __restrict__ sort, const int shift)
{
  const int u = threadIdx.x + blockIdx.x * TPB;
  if (u < g.nodes) {
    const ull val = sort[u];
    g.nindex[u] = val >> shift;
  }
}


static __kernel__ void create_nlist(ECLgraph g, const ull* const __restrict__ sort, const ull mask)
{
  const int e = threadIdx.x + blockIdx.x * TPB;
  if (e < g.edges) {
    const ull val = sort[e];
    g.nlist[e] = val & mask;
  }
}


static __kernel__ void init_wl_triangle(const ECLgraph g, int2* const __restrict__ wl, const int sum1, const int sum2)
{
  const int u = threadIdx.x + blockIdx.x * TPB;
  int u_beg, u_end;
  bool active = false;
  if (u < g.nodes) {
    u_beg = g.nindex[u];
    u_end = g.nindex[u + 1];
    const int u_deg = u_end - u_beg;
    if (u_deg >= sum1) {
      const int thres1 = max(sum1, sum2 - u_deg);
      active = true;
      if (u_deg < WS) {
        active = false;
        for (int i = u_beg; i < u_end; i++) {
          const int v = g.nlist[i];
          if (v >= u) break;  // one direction only
          const int v_beg = g.nindex[v];
          const int v_end = g.nindex[v + 1];
          const int v_deg = v_end - v_beg;
          if (v_deg >= thres1) {
            const int loc = atomicAdd(&wlsize, 1);
            wl[loc] = int2{u, v};
          }
        }
      }
    }
  }
  int bal = __ballot_sync(~0, active);
  const int lane = threadIdx.x % WS;
  while (bal != 0) {
    const int old = bal;
    bal &= bal - 1;
    const int who = __ffs(bal ^ old) - 1;
    const int w_u = __shfl_sync(~0, u, who);
    const int w_u_beg = __shfl_sync(~0, u_beg, who);
    const int w_u_end = __shfl_sync(~0, u_end, who);
    const int w_u_deg = w_u_end - w_u_beg;
    const int w_thres1 = max(sum1, sum2 - w_u_deg);
    for (int i = w_u_beg + lane; i < w_u_end; i += WS) {
      const int v = g.nlist[i];
      if (v >= w_u) break;  // one direction only
      const int v_beg = g.nindex[v];
      const int v_end = g.nindex[v + 1];
      const int v_deg = v_end - v_beg;
      if (v_deg >= w_thres1) {
        const int loc = atomicAdd(&wlsize, 1);
        wl[loc] = int2{w_u, v};
      }
    }
  }
}


static inline void relabelByDegree(const ECLgraph d_g, int* const d_map)
{
  // allocate memory
  ull* d_sort;
  if (cudaSuccess != cudaMalloc((void**)&d_sort, 2 * sizeof(ull) * std::max(d_g.nodes, d_g.edges))) {
    printf("skipping relabeling due to memory constraints\n");
    cudaGetLastError();  // reset to cudaSuccess
  } else {
    ull* d_sort1 = &d_sort[0];
    ull* d_sort2 = &d_sort[std::max(d_g.nodes, d_g.edges)];

    // set up sorting and determine maximum degree
    const int idbits = sizeof(int) * 8 - __builtin_clz(d_g.nodes - 1);
    init_sort<<<(d_g.nodes + TPB - 1) / TPB, TPB>>>(d_g, d_sort1, idbits);
    int d_maxdeg;
    cudaMemcpyFromSymbol(&d_maxdeg, maxdeg, sizeof(maxdeg));
    if (d_maxdeg <= 16) {
      printf("skipping relabeling due to low maximum degree\n");
    } else {
      // determine number of bits
      const int degbits = sizeof(int) * 8 - __builtin_clz(d_maxdeg);

      // sort by degree (descending)
      cub::DoubleBuffer<ull> d_keys1(d_sort1, d_sort2);
      void* d_temp_storage1 = NULL;
      size_t temp_storage_bytes1 = 0;
      cub::DeviceRadixSort::SortKeysDescending(d_temp_storage1, temp_storage_bytes1, d_keys1, d_g.nodes, 0, idbits + degbits);
      cudaMalloc(&d_temp_storage1, temp_storage_bytes1);
      cub::DeviceRadixSort::SortKeysDescending(d_temp_storage1, temp_storage_bytes1, d_keys1, d_g.nodes, 0, idbits + degbits);
      cudaFree(d_temp_storage1);
      if (d_keys1.Current() == d_sort2) std::swap(d_sort1, d_sort2);  // result in d_sort1

      // create map from old to new vertex IDs
      const ull mask = ~(-1LL << idbits);
      create_map<<<(d_g.nodes + TPB - 1) / TPB, TPB>>>(d_g.nodes, d_sort1, d_map, mask);

      // create mapped v#e list
      create_edge_list<<<(d_g.nodes + TPB - 1) / TPB, TPB>>>(d_g, d_map, d_sort2, idbits);

      // create new nindex
      create_nindex<<<(d_g.nodes + TPB - 1) / TPB, TPB>>>(d_g, d_sort1, idbits);

      // compute pfs on nindex
      void* d_temp_storage2 = NULL;
      size_t temp_storage_bytes2 = 0;
      cub::DeviceScan::ExclusiveSum(d_temp_storage2, temp_storage_bytes2, d_g.nindex, d_g.nindex, d_g.nodes + 1);
      cudaMalloc(&d_temp_storage2, temp_storage_bytes2);
      cub::DeviceScan::ExclusiveSum(d_temp_storage2, temp_storage_bytes2, d_g.nindex, d_g.nindex, d_g.nodes + 1);
      cudaFree(d_temp_storage2);

      // sort edge list
      cub::DoubleBuffer<ull> d_keys3(d_sort2, d_sort1);
      void* d_temp_storage3 = NULL;
      size_t temp_storage_bytes3 = 0;
      cub::DeviceRadixSort::SortKeys(d_temp_storage3, temp_storage_bytes3, d_keys3, d_g.edges, 0, idbits + idbits);
      cudaMalloc(&d_temp_storage3, temp_storage_bytes3);
      cub::DeviceRadixSort::SortKeys(d_temp_storage3, temp_storage_bytes3, d_keys3, d_g.edges, 0, idbits + idbits);
      cudaFree(d_temp_storage3);
      if (d_keys3.Current() == d_sort2) std::swap(d_sort1, d_sort2);  // result in d_sort1

      // create new nlist
      create_nlist<<<(d_g.edges + TPB - 1) / TPB, TPB>>>(d_g, d_sort1, mask);
    }

    // clean up
    cudaFree(d_sort);
  }
}


static inline int triangleRotations(const int f [8], int fr [6][8])
{
  int p = 0;
  fr[p][iuvw] = f[iuvw]; fr[p][iuv] = f[iuv]; fr[p][iuw] = f[iuw]; fr[p][ivw] = f[ivw]; fr[p][iu] = f[iu]; fr[p][iv] = f[iv]; fr[p][iw] = f[iw];
  p++;
  if ((f[iu] != f[iv]) || (f[iu] != f[iw]) || (f[iuv] != f[iuw]) || (f[iuv] != f[ivw])) {
    // rotate
    const int q = p - 1;
    fr[p][iuvw] = fr[q][iuvw]; fr[p][iuv] = fr[q][ivw]; fr[p][iuw] = fr[q][iuv]; fr[p][ivw] = fr[q][iuw]; fr[p][iu] = fr[q][iv]; fr[p][iv] = fr[q][iw]; fr[p][iw] = fr[q][iu];
    p++;
    fr[p][iuvw] = fr[q][iuvw]; fr[p][iuv] = fr[q][iuw]; fr[p][iuw] = fr[q][ivw]; fr[p][ivw] = fr[q][iuv]; fr[p][iu] = fr[q][iw]; fr[p][iv] = fr[q][iu]; fr[p][iw] = fr[q][iv];
    p++;
    if (((f[iv] != f[iw]) || (f[iuv] != f[iuw])) && ((f[iu] != f[iw]) || (f[iuv] != f[ivw])) && ((f[iu] != f[iv]) || (f[iuw] != f[ivw]))) {
      // flip
      fr[p][iuvw] = f[iuvw]; fr[p][iuv] = f[iuw]; fr[p][iuw] = f[iuv]; fr[p][ivw] = f[ivw]; fr[p][iu] = f[iu]; fr[p][iv] = f[iw]; fr[p][iw] = f[iv];
      p++;
      // rotate
      const int q = p - 1;
      fr[p][iuvw] = fr[q][iuvw]; fr[p][iuv] = fr[q][ivw]; fr[p][iuw] = fr[q][iuv]; fr[p][ivw] = fr[q][iuw]; fr[p][iu] = fr[q][iv]; fr[p][iv] = fr[q][iw]; fr[p][iw] = fr[q][iu];
      p++;
      fr[p][iuvw] = fr[q][iuvw]; fr[p][iuv] = fr[q][iuw]; fr[p][iuw] = fr[q][ivw]; fr[p][ivw] = fr[q][iuv]; fr[p][iu] = fr[q][iw]; fr[p][iv] = fr[q][iu]; fr[p][iw] = fr[q][iv];
      p++;
    }
  }

  return p;
}


static ull triangleCore(const MatchingOrder mo, const ECLgraph g, int2* const wl, const int SMs, const int mTpSM)
{
  int f [8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < mo.anchorsets; i++) {
    const int as = mo.anch[i];
    f[as] = mo.fcnt[i];
  }

  // pre-compute minimum degrees
  const int min_u = f[iuvw] + f[iuv] + f[iuw] + f[iu];
  const int min_v = f[iuvw] + f[iuv] + f[ivw] + f[iv];
  const int min_w = f[iuvw] + f[iuw] + f[ivw] + f[iw];
  const int sum3 = min_u + min_v + min_w + 6;
  const int sum2 = sum3 - (std::max(std::max(min_u, min_v), min_w) + 2);
  const int sum1 = std::min(std::min(min_u, min_v), min_w) + 2;

  // determine needed flips and rotations
  int fr [6][8];
  const int rot = triangleRotations(f, fr);
  cudaMemcpyToSymbol(d_fr, fr, sizeof(int) * 8 * 6);

  // optimize graph vertex labeling
  if (g.nodes > 10000) {
    if (g.nodes > g.edges / 10) {
      printf("skipping relabeling due to low average degree\n");
    } else {
      CPUTimer timer;
      timer.start();
      relabelByDegree(g, (int*)wl);  // reusing memory
      const double relabeltime = timer.elapsed();
      printf("relabel time: %.6f s\n", relabeltime);
    }
  }
  fflush(stdout);

  // set up worklist
  init_wl_triangle<<<(g.nodes + TPB - 1) / TPB, TPB>>>(g, wl, sum1, sum2);

  // count occurrences
  const int blocks = SMs * (mTpSM / TPB);
  if ((f[iuvw] == 1) && (f[iuv] + f[iuw] + f[ivw] == 0) && ((f[iu] == 0) + (f[iv] == 0) + (f[iw] == 0) >= 2)) {
    if (f[iu] + f[iv] + f[iw] == 0) {
      clique<<<blocks, TPB>>>(g, wl);
    } else {
      tailed_clique_triangle<<<blocks, TPB>>>(g, wl, f[iu] + f[iv] + f[iw]);
    }
  } else {
    general_triangle_core<<<blocks, TPB>>>(g, wl, rot, sum1, sum2, sum3);
  }
  ull count;
  cudaMemcpyFromSymbol(&count, total, sizeof(total));

  return count;
}


/**************************************************************/
/* 3-vertex wedge-core code                                   */
/**************************************************************/


static inline __device__ ull wedge(const int u_beg, const int u_end, const int v_beg, const int v_end, const int* const __restrict__ nlist, const int f_u, const int f_uv, const int f_v, const int c_uv, const int u)
{
  const int a_uv = c_uv - f_uv;
  if (a_uv < 0) return 0;

  const int corr = contains(u, v_beg, v_end, nlist) ? 2 : 1;
  const int u_deg = u_end - u_beg;
  const int v_deg = v_end - v_beg;
  const int a_u = u_deg - (corr + c_uv);
  const int a_v = v_deg - (corr + c_uv);
  const int bound = min(f_u, a_uv);
  const int a_uv_v = a_uv + a_v;
  const int a_u_uv = a_uv + a_u;

  ull sum = 0;
  for (int j = 0; j <= bound; j++) {
    const int k = f_u - j;
    ull term = nCk(a_u, k) * nCk(a_uv_v - j, f_v);
    if (f_u != f_v) {  // asymmetric
      term += nCk(a_v, k) * nCk(a_u_uv - j, f_v);
    }
    sum += term * nCk(a_uv, j);
  }

  return sum * nCk(c_uv, f_uv);
}


static __kernel__ void wedgecore_specialized(const ECLgraph g, const int2* const __restrict__ wl, const int f_v, const int f_vw, const int f_w)
{
  // pre-compute minimum degrees
  const int min_v = f_v + f_vw + 1;
  const int min_w = f_vw + f_w + 1;
  const int lim = min(min_v, min_w);
  const int lane = threadIdx.x % WS;

  ull sum = 0;
  do {
    const int i = atomicAdd(&wlpos, 1);
    bool active = (i < wlsize);
    if (__all_sync(~0, !active)) break;

    int v, u_beg, u_end, v_beg, v_end;
    if (active) {
      const int2 ele = wl[i];  // u, v
      const int u = ele.x;
      v = ele.y;
      u_beg = g.nindex[u];
      u_end = g.nindex[u + 1];
      v_beg = g.nindex[v];
      v_end = g.nindex[v + 1];
    }

    int bal = __ballot_sync(~0, active);
    while (bal != 0) {
      const int old = bal;
      bal &= bal - 1;
      const int who = __ffs(bal ^ old) - 1;
      const int w_v = __shfl_sync(~0, v, who);
      const int w_u_beg = __shfl_sync(~0, u_beg, who);
      const int w_u_end = __shfl_sync(~0, u_end, who);
      const int w_v_beg = __shfl_sync(~0, v_beg, who);
      const int w_v_end = __shfl_sync(~0, v_end, who);
      const int w_v_deg = w_v_end - w_v_beg;
      const int thres = max(lim, min_v + min_w - w_v_deg);
      for (int j = w_u_beg + lane; __any_sync(~0, j < w_u_end); j += WS) {
        bool active = (j < w_u_end);
        int w, w_beg, w_end, w_deg;
        if (active) {
          w = g.nlist[j];
          active = (w < w_v);  // w < v
          if (active) {
            w_beg = g.nindex[w];
            w_end = g.nindex[w + 1];
            w_deg = w_end - w_beg;
            active = (w_deg >= thres);
          }
        }
        int c_uv, bal = __ballot_sync(~0, active);
        while (bal != 0) {
          const int old = bal;
          bal &= bal - 1;
          const int who = __ffs(bal ^ old) - 1;
          const int w_w_beg = __shfl_sync(~0, w_beg, who);
          const int w_w_end = __shfl_sync(~0, w_end, who);
          const int cnt = common2(w_v_beg, w_v_end, w_w_beg, w_w_end, g.nlist);
          if (who == lane) c_uv = cnt - 1;
        }
        if (active) {
          sum += wedge(w_v_beg, w_v_end, w_beg, w_end, g.nlist, f_v, f_vw, f_w, c_uv, w_v);
        }
      }
    }
  } while (true);

  atomicAdd(&total, sum);
}


static inline __device__ void common3_wedge(const int beg1, const int end1, const int beg2, const int end2, const int beg3, const int end3, const int* const __restrict__ nlist, int venn [8], const int p12)  // warp-based
{
  const int lane = threadIdx.x % WS;
  int t13 = 0;
  int t23 = 0;
  int t123 = 0;
  for (int j = beg3 + lane; j < end3; j += WS) {
    const int q = nlist[j];
    const bool b13 = contains(q, beg1, end1, nlist);
    const bool b23 = contains(q, beg2, end2, nlist);
    t13 += b13;
    t23 += b23;
    t123 += b13 & b23;
  }
  const int c123 = __reduce_add_sync(~0, t123);
  const int c12 = p12 - c123;
  const int c13 = __reduce_add_sync(~0, t13) - c123;
  const int c23 = __reduce_add_sync(~0, t23) - c123;
  const int c1 = end1 - beg1 - (p12 + c13);
  const int c2 = end2 - beg2 - (p12 + c23);
  const int c3 = end3 - beg3 - (c13 + c23 + c123);
  if (lane == 0) {
    venn[iu] = c1;
    venn[iv] = c2;
    venn[iw] = c3;
    venn[iuv] = c12;
    venn[iuw] = c13;
    venn[ivw] = c23;
    venn[iuvw] = c123;
  }
}


static __global__
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
  __launch_bounds__(TPB, 1536 / TPB)
#else
  __launch_bounds__(TPB, 1024 / TPB)
#endif
void general_wedge_core(const ECLgraph g, const int2* const __restrict__ wl, const int min_u, const int lim, const int sum1, const int sum2, const int sum3, const int rotations)
{
  const int lane = threadIdx.x % WS;
  const int warpoffs = (threadIdx.x / WS) * WS;
  __shared__ int venn [TPB * 8];

  ull sum = 0;
  do {
    const int i = atomicAdd(&wlpos, 1);
    bool active = (i < wlsize);
    if (__all_sync(~0, !active)) break;

    int u, v, u_beg, u_end, v_beg, v_end;
    if (active) {
      const int2 ele = wl[i];  // u, v
      u = ele.x;
      v = ele.y;
      u_beg = g.nindex[u];
      u_end = g.nindex[u + 1];
      v_beg = g.nindex[v];
      v_end = g.nindex[v + 1];
    }

    int bal = __ballot_sync(~0, active);
    while (bal != 0) {
      const int old = bal;
      bal &= bal - 1;
      const int who = __ffs(bal ^ old) - 1;
      const int w_u = __shfl_sync(~0, u, who);
      const int w_v = __shfl_sync(~0, v, who);
      const int w_u_beg = __shfl_sync(~0, u_beg, who);
      const int w_u_end = __shfl_sync(~0, u_end, who);
      const int w_v_beg = __shfl_sync(~0, v_beg, who);
      const int w_v_end = __shfl_sync(~0, v_end, who);
      const int w_u_deg = w_u_end - w_u_beg;
      const int w_v_deg = w_v_end - w_v_beg;

      const int p12 = common2(w_u_beg, w_u_end, w_v_beg, w_v_end, g.nlist);

      const bool cond = (w_u_deg >= min_u) && (w_v_deg >= lim);
      const int thres1 = max(lim, sum2 - w_v_deg);
      const int thres2 = max(max(sum1, sum3 - w_u_deg - w_v_deg), max(sum2 - w_u_deg, sum2 - w_v_deg));

      for (int j = w_u_beg + lane; __any_sync(~0, j < w_u_end); j += WS) {
        bool tri, active = (j < w_u_end);
        int w, w_beg, w_end, w_deg;
        if (active) {
          w = g.nlist[j];
          active = (w < w_v);  // w < v
          if (active) {
            w_beg = g.nindex[w];
            w_end = g.nindex[w + 1];
            w_deg = w_end - w_beg;
            active = (w_deg >= thres2);
            if (active) {
              tri = contains(w, w_v_beg, w_v_end, g.nlist);
              if (tri) {
                active = (w_u < w);
              } else {
                active = cond && (w_deg >= thres1);
              }
            }
          }
        }

        // wedge with u in center
        int bal = __ballot_sync(~0, active & !tri);
        int cnt = 0;
        while (bal != 0) {
          const int old = bal;
          bal &= bal - 1;
          const int who = __ffs(bal ^ old) - 1;
          const int w_w_beg = __shfl_sync(~0, w_beg, who);
          const int w_w_end = __shfl_sync(~0, w_end, who);
          common3_wedge(w_u_beg, w_u_end, w_v_beg, w_v_end, w_w_beg, w_w_end, g.nlist, &venn[(warpoffs + cnt) * 8], p12);
          cnt++;
        }
        __syncwarp();

        if (lane < cnt) {
          //correct venn: -2 -1 -1:   a_vw -1  a_u -2
          venn[(warpoffs + lane) * 8 + iu] -= 2;
          venn[(warpoffs + lane) * 8 + ivw]--;
        }
        __syncwarp();

        if (rotations > 3) {  // asymmetric
          for (int i = lane; i < 2 * cnt; i += WS) {
            sum += fc3(d_fr[(i % 2) * 3], &venn[(warpoffs + i / 2) * 8]);
          }
        } else {
          if (lane < cnt) {
            sum += fc3(d_fr[0], &venn[(warpoffs + lane) * 8]);
          }
        }
        __syncwarp();

        // triangle (3 wedges)
        bal = __ballot_sync(~0, active & tri);
        cnt = 0;
        while (bal != 0) {
          const int old = bal;
          bal &= bal - 1;
          const int who = __ffs(bal ^ old) - 1;
          const int w_w_beg = __shfl_sync(~0, w_beg, who);
          const int w_w_end = __shfl_sync(~0, w_end, who);
          const bool w_tri = __shfl_sync(~0, tri, who);
          common3_wedge(w_u_beg, w_u_end, w_v_beg, w_v_end, w_w_beg, w_w_end, g.nlist, &venn[(warpoffs + cnt) * 8], p12);
          cnt++;
        }
        __syncwarp();

        if (lane < cnt) {
          //correct venn: -2 -2 -2:   a_uv -1  a_uw -1  a_vw -1
          venn[(warpoffs + lane) * 8 + iuv]--;
          venn[(warpoffs + lane) * 8 + iuw]--;
          venn[(warpoffs + lane) * 8 + ivw]--;
        }
        __syncwarp();

        const int tot = rotations * cnt;
        for (int i = lane; i < tot; i += WS) {
          const int r = i / cnt;
          const int p = i % cnt;
          sum += fc3(d_fr[r], &venn[(warpoffs + p) * 8]);
        }
      }
    }
  } while (true);

  atomicAdd(&total, sum);
}


static __kernel__ void init_wl_specialized_wedge(const ECLgraph g, int2* const __restrict__ wl, const int lim)
{
  const int u = threadIdx.x + blockIdx.x * TPB;
  int u_beg, u_end;
  bool active = false;
  if (u < g.nodes) {
    u_beg = g.nindex[u];
    u_end = g.nindex[u + 1];
    const int u_deg = u_end - u_beg;
    if (u_deg >= 2) {
      active = true;
      if (u_deg < WS) {
        active = false;
        for (int i = u_beg; i < u_end; i++) {
          const int v = g.nlist[i];
          const int v_beg = g.nindex[v];
          const int v_end = g.nindex[v + 1];
          const int v_deg = v_end - v_beg;
          if (v_deg >= lim) {
            const int loc = atomicAdd(&wlsize, 1);
            wl[loc] = int2{u, v};
          }
        }
      }
    }
  }
  int bal = __ballot_sync(~0, active);
  const int lane = threadIdx.x % WS;
  while (bal != 0) {
    const int old = bal;
    bal &= bal - 1;
    const int who = __ffs(bal ^ old) - 1;
    const int w_u = __shfl_sync(~0, u, who);
    const int w_u_beg = __shfl_sync(~0, u_beg, who);
    const int w_u_end = __shfl_sync(~0, u_end, who);
    for (int i = w_u_beg + lane; i < w_u_end; i += WS) {
      const int v = g.nlist[i];
      const int v_beg = g.nindex[v];
      const int v_end = g.nindex[v + 1];
      const int v_deg = v_end - v_beg;
      if (v_deg >= lim) {
        const int loc = atomicAdd(&wlsize, 1);
        wl[loc] = int2{w_u, v};
      }
    }
  }
}


static __kernel__ void init_wl_wedge(const ECLgraph g, int2* const __restrict__ wl, const int sum1, const int sum2)
{
  const int u = threadIdx.x + blockIdx.x * TPB;
  int u_beg, u_end;
  bool active = false;
  if (u < g.nodes) {
    u_beg = g.nindex[u];
    u_end = g.nindex[u + 1];
    const int u_deg = u_end - u_beg;
    if (u_deg >= sum1) {
      const int thres1 = max(sum1, sum2 - u_deg);
      active = true;
      if (u_deg < WS) {
        active = false;
        for (int i = u_beg; i < u_end; i++) {
          const int v = g.nlist[i];
          const int v_beg = g.nindex[v];
          const int v_end = g.nindex[v + 1];
          const int v_deg = v_end - v_beg;
          if (v_deg >= thres1) {
            const int loc = atomicAdd(&wlsize, 1);
            wl[loc] = int2{u, v};
          }
        }
      }
    }
  }
  int bal = __ballot_sync(~0, active);
  const int lane = threadIdx.x % WS;
  while (bal != 0) {
    const int old = bal;
    bal &= bal - 1;
    const int who = __ffs(bal ^ old) - 1;
    const int w_u = __shfl_sync(~0, u, who);
    const int w_u_beg = __shfl_sync(~0, u_beg, who);
    const int w_u_end = __shfl_sync(~0, u_end, who);
    const int w_u_deg = w_u_end - w_u_beg;
    const int w_thres1 = max(sum1, sum2 - w_u_deg);
    for (int i = w_u_beg + lane; i < w_u_end; i += WS) {
      const int v = g.nlist[i];
      const int v_beg = g.nindex[v];
      const int v_end = g.nindex[v + 1];
      const int v_deg = v_end - v_beg;
      if (v_deg >= w_thres1) {
        const int loc = atomicAdd(&wlsize, 1);
        wl[loc] = int2{w_u, v};
      }
    }
  }
}


static inline int wedgeRotations(const int f [8], int fr [6][8])
{
  int p = 0;
  fr[p][iuvw] = f[iuvw]; fr[p][iuv] = f[iuv]; fr[p][iuw] = f[iuw]; fr[p][ivw] = f[ivw]; fr[p][iu] = f[iu]; fr[p][iv] = f[iv]; fr[p][iw] = f[iw];
  p++;
  if (true) {
    // rotate
    const int q = p - 1;
    fr[p][iuvw] = fr[q][iuvw]; fr[p][iuv] = fr[q][ivw]; fr[p][iuw] = fr[q][iuv]; fr[p][ivw] = fr[q][iuw]; fr[p][iu] = fr[q][iv]; fr[p][iv] = fr[q][iw]; fr[p][iw] = fr[q][iu];
    p++;
    fr[p][iuvw] = fr[q][iuvw]; fr[p][iuv] = fr[q][iuw]; fr[p][iuw] = fr[q][ivw]; fr[p][ivw] = fr[q][iuv]; fr[p][iu] = fr[q][iw]; fr[p][iv] = fr[q][iu]; fr[p][iw] = fr[q][iv];
    p++;
    if ((f[iv] != f[iw]) || (f[iuv] != f[iuw])) {
      // flip
      fr[p][iuvw] = f[iuvw]; fr[p][iuv] = f[iuw]; fr[p][iuw] = f[iuv]; fr[p][ivw] = f[ivw]; fr[p][iu] = f[iu]; fr[p][iv] = f[iw]; fr[p][iw] = f[iv];
      p++;
      // rotate
      const int q = p - 1;
      fr[p][iuvw] = fr[q][iuvw]; fr[p][iuv] = fr[q][ivw]; fr[p][iuw] = fr[q][iuv]; fr[p][ivw] = fr[q][iuw]; fr[p][iu] = fr[q][iv]; fr[p][iv] = fr[q][iw]; fr[p][iw] = fr[q][iu];
      p++;
      fr[p][iuvw] = fr[q][iuvw]; fr[p][iuv] = fr[q][iuw]; fr[p][iuw] = fr[q][ivw]; fr[p][ivw] = fr[q][iuv]; fr[p][iu] = fr[q][iw]; fr[p][iv] = fr[q][iu]; fr[p][iw] = fr[q][iv];
      p++;
    }
  }

  return p;
}


static ull wedgeCore(const MatchingOrder mo, const ECLgraph g, int2* const wl, const int SMs, const int mTpSM)
{
  int f [8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < mo.anchorsets; i++) {
    const int as = mo.anch[i];
    f[as] = mo.fcnt[i];
  }

  if (mo.list[1] == 2) {
    // swap u <-> v so u is center of wedge
    std::swap(f[iu], f[iv]);
    std::swap(f[iuw], f[ivw]);
  }

  // pre-compute minimum degrees
  const int min_u = f[iuvw] + f[iuv] + f[iuw] + f[iu] + 2;
  const int min_v = f[iuvw] + f[iuv] + f[ivw] + f[iv] + 1;
  const int min_w = f[iuvw] + f[iuw] + f[ivw] + f[iw] + 1;
  const int sum3 = min_u + min_v + min_w;
  const int sum2 = sum3 - std::max(std::max(min_u, min_v), min_w);
  const int sum1 = std::min(std::min(min_u, min_v), min_w);
  const int lim = std::min(min_v, min_w);

  if ((f[iuvw] == 0) && (f[iuv] == 0) && (f[iuw] == 0) && (f[iu] == 0)) {
    // special case where center u of wedge has no fringes
    init_wl_specialized_wedge<<<(g.nodes + TPB - 1) / TPB, TPB>>>(g, wl, lim);
    const int blocks = SMs * (mTpSM / TPB);
    wedgecore_specialized<<<blocks, TPB>>>(g, wl, f[iv], f[ivw], f[iw]);
  } else {
    // determine needed flips and rotations
    int fr [6][8];
    const int rot = wedgeRotations(f, fr);
    cudaMemcpyToSymbol(d_fr, fr, sizeof(int) * 8 * 6);

    init_wl_wedge<<<(g.nodes + TPB - 1) / TPB, TPB>>>(g, wl, sum1, sum2);
    const int blocks = SMs * ((mTpSM - 512) / TPB);
    general_wedge_core<<<blocks, TPB>>>(g, wl, min_u, lim, sum1, sum2, sum3, rot);
  }
  ull count;
  cudaMemcpyFromSymbol(&count, total, sizeof(total));

  return count;
}


/**************************************************************/
/* 2-vertex edge-core code                                    */
/**************************************************************/


static inline __device__ int common_triangle(const int beg1, const int end1, const int beg2, const int end2, const int* const __restrict__ nlist, const int top)
{
  const int deg1 = end1 - beg1;
  const int deg2 = end2 - beg2;
  const int lane = threadIdx.x % WS;
  const int beg = (deg1 < deg2) ? beg1 : beg2;
  const int end = (deg1 < deg2) ? end1 : end2;
  const int left = (deg1 < deg2) ? beg2 : beg1;
  const int right = (deg1 < deg2) ? end2 : end1;
  int c12 = 0;
  const int middle = (left + right) / 2;
  for (int i = beg + lane; i < end; i += WS) {
    const int w = nlist[i];
    if (w >= top) break;
    int l = left;
    int r = right;
    int m = middle;
    while (m != l) {
      if (w < nlist[m]) {
        r = m;
      } else {
        l = m;
      }
      m = (l + r) / 2;
    }
    if (w == nlist[m]) c12++;
  }
  return c12;
}


static __kernel__ void triangle(const ECLgraph g, const int2* const __restrict__ wl)
{
  const int i = threadIdx.x + blockIdx.x * TPB;
  bool active = (i < wlsize);
  int v, u_beg, u_end, v_beg, v_end;
  if (active) {
    const int2 ele = wl[i];  // u, v
    const int u = ele.x;
    v = ele.y;
    u_beg = g.nindex[u];
    u_end = g.nindex[u + 1];
    v_beg = g.nindex[v];
    v_end = g.nindex[v + 1];
  }

  int bal = __ballot_sync(~0, active);
  ull cnt = 0;
  while (bal != 0) {
    const int old = bal;
    bal &= bal - 1;
    const int who = __ffs(bal ^ old) - 1;
    const int w_v = __shfl_sync(~0, v, who);
    const int w_u_beg = __shfl_sync(~0, u_beg, who);
    const int w_u_end = __shfl_sync(~0, u_end, who);
    const int w_v_beg = __shfl_sync(~0, v_beg, who);
    const int w_v_end = __shfl_sync(~0, v_end, who);
    cnt += common_triangle(w_u_beg, w_u_end, w_v_beg, w_v_end, g.nlist, w_v);
  }

  atomicAdd(&total, cnt);
}


static inline __device__ ull common_tailed_triangle(const int beg1, const int end1, const int beg2, const int end2, const int* const __restrict__ nindex, const int* const __restrict__ nlist, const int top, const int tails, const ull c12)
{
  const int deg1 = end1 - beg1;
  const int deg2 = end2 - beg2;
  const int lane = threadIdx.x % WS;
  const int beg = (deg1 < deg2) ? beg1 : beg2;
  const int end = (deg1 < deg2) ? end1 : end2;
  const int left = (deg1 < deg2) ? beg2 : beg1;
  const int right = (deg1 < deg2) ? end2 : end1;
  ull total = 0;
  const int middle = (left + right) / 2;
  for (int i = beg + lane; i < end; i += WS) {
    const int w = nlist[i];
    if (w >= top) break;
    int l = left;
    int r = right;
    int m = middle;
    while (m != l) {
      if (w < nlist[m]) {
        r = m;
      } else {
        l = m;
      }
      m = (l + r) / 2;
    }
    if (w == nlist[m]) {
      const int deg = nindex[w + 1] - nindex[w];
      total += c12 + nCk(deg - 2, tails);
    }
  }
  return total;
}


static __kernel__ void tailed_triangle(const ECLgraph g, const int2* const __restrict__ wl, const int tails)
{
  const int i = threadIdx.x + blockIdx.x * TPB;
  bool active = (i < wlsize);
  int v, u_beg, u_end, v_beg, v_end;
  ull c12;
  if (active) {
    const int2 ele = wl[i];  // u, v
    const int u = ele.x;
    v = ele.y;
    u_beg = g.nindex[u];
    u_end = g.nindex[u + 1];
    v_beg = g.nindex[v];
    v_end = g.nindex[v + 1];
    const ull c1 = nCk(u_end - u_beg - 2, tails);
    const ull c2 = nCk(v_end - v_beg - 2, tails);
    c12 = c1 + c2;
  }

  int bal = __ballot_sync(~0, active);
  ull cnt = 0;
  while (bal != 0) {
    const int old = bal;
    bal &= bal - 1;
    const int who = __ffs(bal ^ old) - 1;
    const int w_v = __shfl_sync(~0, v, who);
    const int w_u_beg = __shfl_sync(~0, u_beg, who);
    const int w_u_end = __shfl_sync(~0, u_end, who);
    const int w_v_beg = __shfl_sync(~0, v_beg, who);
    const int w_v_end = __shfl_sync(~0, v_end, who);
    const ull w_c12 = __shfl_sync(~0, c12, who);
    cnt += common_tailed_triangle(w_u_beg, w_u_end, w_v_beg, w_v_end, g.nindex, g.nlist, w_v, tails, w_c12);
  }

  atomicAdd(&total, cnt);
}


static inline __device__ int common(const int beg1, const int end1, const int beg2, const int end2, const int* const __restrict__ nlist)
{
  const int deg1 = end1 - beg1;
  const int deg2 = end2 - beg2;
  const int lane = threadIdx.x % WS;
  const int beg = (deg1 < deg2) ? beg1 : beg2;
  const int end = (deg1 < deg2) ? end1 : end2;
  const int left = (deg1 < deg2) ? beg2 : beg1;
  const int right = (deg1 < deg2) ? end2 : end1;
  int c12 = 0;
  const int middle = (left + right) / 2;
  for (int i = beg + lane; i < end; i += WS) {
    const int w = nlist[i];
    int l = left;
    int r = right;
    int m = middle;
    while (m != l) {
      if (w < nlist[m]) {
        r = m;
      } else {
        l = m;
      }
      m = (l + r) / 2;
    }
    if (w == nlist[m]) c12++;
  }
  return c12;
}


static __kernel__ void general_edge_core(const ECLgraph g, const int2* const __restrict__ wl, const int f_u, const int f_uv, const int f_v)
{
  const int i = threadIdx.x + blockIdx.x * TPB;
  const int lane = threadIdx.x % WS;
  bool active = (i < wlsize);
  int u_beg, u_end, v_beg, v_end, c_uv = -1;
  if (active) {
    c_uv = 0;
    const int2 ele = wl[i];  // u, v
    const int u = ele.x;
    const int v = ele.y;
    u_beg = g.nindex[u];
    u_end = g.nindex[u + 1];
    v_beg = g.nindex[v];
    v_end = g.nindex[v + 1];
  }

  int bal = __ballot_sync(~0, active);
  while (bal != 0) {
    const int old = bal;
    bal &= bal - 1;
    const int who = __ffs(bal ^ old) - 1;
    const int w_u_beg = __shfl_sync(~0, u_beg, who);
    const int w_u_end = __shfl_sync(~0, u_end, who);
    const int w_v_beg = __shfl_sync(~0, v_beg, who);
    const int w_v_end = __shfl_sync(~0, v_end, who);
    const int part = common(w_u_beg, w_u_end, w_v_beg, w_v_end, g.nlist);
    const int red = __reduce_add_sync(~0, part);
    if (lane == who) c_uv = red;
  }

  const int a_uv = c_uv - f_uv;
  if (a_uv >= 0) {
    const int u_deg = u_end - u_beg;
    const int v_deg = v_end - v_beg;
    const int c_u = u_deg - 1 - c_uv;
    const int c_v = v_deg - 1 - c_uv;
    const int bound = min(f_u, a_uv);
    const int a_uv_v = a_uv + c_v;
    const int a_u_uv = a_uv + c_u;
    ull acc = 0;
    for (int j = 0; j <= bound; j++) {
      const int k = f_u - j;
      ull term = nCk(c_u, k) * nCk(a_uv_v - j, f_v);
      if (f_u != f_v) {  // asymmetric
        term += nCk(c_v, k) * nCk(a_u_uv - j, f_v);
      }
      acc += term * nCk(a_uv, j);
    }
    const ull cnt = acc * nCk(c_uv, f_uv);
    atomicAdd(&total, cnt);
  }
}


static __kernel__ void init_wl_edge(const ECLgraph g, int2* const __restrict__ wl, const int f_u_uv, const int f_u_uv_v)
{
  const int u = threadIdx.x + blockIdx.x * TPB;
  int u_beg, u_end;
  bool active = false;
  if (u < g.nodes) {
    u_beg = g.nindex[u];
    u_end = g.nindex[u + 1];
    const int u_deg = u_end - u_beg;
    if (u_deg > f_u_uv) {
      active = true;
      if (u_deg < WS) {
        active = false;
        for (int i = u_beg; i < u_end; i++) {
          const int v = g.nlist[i];
          if (v >= u) break;  // one direction only
          const int v_beg = g.nindex[v];
          const int v_end = g.nindex[v + 1];
          const int v_deg = v_end - v_beg;
          if ((v_deg > f_u_uv) && (u_deg + v_deg > f_u_uv_v)) {
            const int loc = atomicAdd(&wlsize, 1);
            wl[loc] = int2{u, v};
          }
        }
      }
    }
  }
  int bal = __ballot_sync(~0, active);
  const int lane = threadIdx.x % WS;
  while (bal != 0) {
    const int old = bal;
    bal &= bal - 1;
    const int who = __ffs(bal ^ old) - 1;
    const int w_u = __shfl_sync(~0, u, who);
    const int w_u_beg = __shfl_sync(~0, u_beg, who);
    const int w_u_end = __shfl_sync(~0, u_end, who);
    const int w_u_deg = w_u_end - w_u_beg;
    for (int i = w_u_beg + lane; i < w_u_end; i += WS) {
      const int v = g.nlist[i];
      if (v >= w_u) break;  // one direction only
      const int v_beg = g.nindex[v];
      const int v_end = g.nindex[v + 1];
      const int v_deg = v_end - v_beg;
      if ((v_deg > f_u_uv) && (w_u_deg + v_deg > f_u_uv_v)) {
        const int loc = atomicAdd(&wlsize, 1);
        wl[loc] = int2{w_u, v};
      }
    }
  }
}


static ull edgeCore(const MatchingOrder mo, const ECLgraph g, int2* const wl)
{
  int f [4] = {0, 0, 0, 0};
  for (int i = 0; i < mo.anchorsets; i++) {
    const int as = mo.anch[i];
    f[as] = mo.fcnt[i];
  }

  const int f_u = std::min(f[1], f[2]);
  const int f_uv = f[3];
  const int f_v = std::max(f[1], f[2]);

  // set up worklist
  if ((f_uv == 1) && ((f_u == 0) || (f_v == 0))) {
    init_wl_edge<<<(g.nodes + TPB - 1) / TPB, TPB>>>(g, wl, 1, 1);
  } else {
    init_wl_edge<<<(g.nodes + TPB - 1) / TPB, TPB>>>(g, wl, f_u + f_uv, f_u + f_uv + f_v);
  }
  int d_wlsize;
  cudaMemcpyFromSymbol(&d_wlsize, wlsize, sizeof(wlsize));
  const int blocks = (d_wlsize + TPB - 1) / TPB;

  ull count = 0;
  if (blocks > 0) {
    cudaMemcpyToSymbol(total, &count, sizeof(total));
    if ((f_uv == 1) && ((f_u == 0) || (f_v == 0))) {
      if ((f_u == 0) && (f_v == 0)) {
        triangle<<<blocks, TPB>>>(g, wl);
      } else {
        tailed_triangle<<<blocks, TPB>>>(g, wl, f_v);
      }
    } else {
      general_edge_core<<<blocks, TPB>>>(g, wl, f_u, f_uv, f_v);
    }
    cudaMemcpyFromSymbol(&count, total, sizeof(total));
  }

  return count;
}


/**************************************************************/
/* 1-vertex vertex-core code                                  */
/**************************************************************/


static __kernel__ void k_stars(const ECLgraph g, const int tails)
{
  const int u = threadIdx.x + blockIdx.x * TPB;
  if (u < g.nodes) {
    const int u_beg = g.nindex[u];
    const int u_end = g.nindex[u + 1];
    const int n = u_end - u_beg;
    if (n >= tails) {
      ull cnt = 1;
      if (n != tails) {
        const int k = min(tails, n - tails);
        cnt = n;
        for (int i = 2; i <= k; i++) cnt = cnt * (n + 1 - i) / i;
      }
      atomicAdd(&total, cnt);
    }
  }
}


static ull vertexCore(const MatchingOrder mo, const ECLgraph g)
{
  // single vertex
  if (mo.anchorsets == 0) {
    return g.nodes;
  }

  // single edge
  if (mo.fcnt[0] == 1) {
    return g.edges / 2;
  }

  // star
  ull count = 0;
  cudaMemcpyToSymbol(total, &count, sizeof(total));
  const int tails = mo.fcnt[0];
  k_stars<<<(g.nodes + TPB - 1) / TPB, TPB>>>(g, tails);
  cudaMemcpyFromSymbol(&count, total, sizeof(total));
  return count;
}


/**************************************************************/
/* main code                                                  */
/**************************************************************/


static ull occurrences(const MatchingOrder mo, const ECLgraph g, int2* const wl, const int SMs, const int mTpSM, const bool doauto = true)
{
  // 1-vertex core
  if (mo.nodes == 1) {
printf("vertex core\n");
    return vertexCore(mo, g);
  }

  // 2-vertex core
  if (mo.nodes == 2) {
printf("edge core\n");
    return edgeCore(mo, g, wl);
  }

  int automorphisms = 1;
  if (doauto) {
    ECLgraph mg = matchingOrder2ECLgraph(mo);
    if ((mg.nodes > g.nodes) || (mg.edges > g.edges)) {
      freeECLgraph(mg);
      return 0;
    }

    // copy graph to GPU
    ECLgraph d_mg = mg;
    cudaMalloc((void**)&d_mg.nindex, sizeof(int) * (mg.nodes + 1));
    cudaMalloc((void**)&d_mg.nlist, sizeof(int) * mg.edges);
    cudaMemcpy(d_mg.nindex, mg.nindex, sizeof(int) * (mg.nodes + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mg.nlist, mg.nlist, sizeof(int) * mg.edges, cudaMemcpyHostToDevice);
    CheckCuda(__LINE__);

    // count automorphisms
    automorphisms = occurrences(mo, d_mg, wl, SMs, mTpSM, false);
    printf("automorphisms: %d\n", automorphisms);
    assert(automorphisms > 0);

    // reset
    ull dummy = 0;
    cudaMemcpyToSymbol(total, &dummy, sizeof(ull));
    cudaMemcpyToSymbol(wlsize, &dummy, sizeof(int));
    cudaMemcpyToSymbol(wlpos, &dummy, sizeof(int));
    cudaMemcpyToSymbol(maxdeg, &dummy, sizeof(int));

    // clean up
    freeECLgraph(mg);
    cudaFree(d_mg.nindex);
    cudaFree(d_mg.nlist);
    CheckCuda(__LINE__);
  }

  // 3-vertex core
  if (mo.nodes == 3) {
    if (__builtin_popcount(mo.list[1]) == 1) {
if (!doauto) printf("wedge core\n");
      return wedgeCore(mo, g, wl, SMs, mTpSM) / automorphisms;
    } else {
if (!doauto) printf("triangle core\n");
      return triangleCore(mo, g, wl, SMs, mTpSM) / automorphisms;
    }
  }

  // larger cores
if (!doauto) printf("general core\n");
  return generalCore(mo, g, wl, SMs, mTpSM) / automorphisms;
}


int main(int argc, char* argv [])
{
  printf("GPU Fringe Counting v0.10 (%s)\n", __FILE__);
  printf("Copyright 2025 Texas State University\n\n");

  if (argc != 3) {
    printf("USAGE: %s input_graph motif_matching_order\n", argv[0]);
    return -1;
  }

  // get device properties
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: no CUDA capable device detected\n\n"); return -1;}
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  printf("GPU: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n", deviceProp.name, SMs, mTpSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);
  printf("     %.1f GB/s peak bandwidth (%d-bit bus), compute capability %d.%d\n\n", 2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) * 0.000001, deviceProp.memoryBusWidth, deviceProp.major, deviceProp.minor);

  // read graph
  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d (%d)\n\n", g.edges / 2, g.edges);

  // copy graph to GPU
  ECLgraph d_g = g;
  cudaMalloc((void**)&d_g.nindex, sizeof(int) * (g.nodes + 1));
  cudaMalloc((void**)&d_g.nlist, sizeof(int) * g.edges);
  cudaMemcpy(d_g.nindex, g.nindex, sizeof(int) * (g.nodes + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g.nlist, g.nlist, sizeof(int) * g.edges, cudaMemcpyHostToDevice);
  CheckCuda(__LINE__);

  // read matching order
  MatchingOrder mo;
  FILE* f = fopen(argv[2], "rb");
  const int size = fread(&mo, 1, sizeof(MatchingOrder), f);
  fclose(f);
  if (size != sizeof(MatchingOrder)) {
    fprintf(stderr, "ERROR: could not read matching order from file\n");
    return -1;
  }

  printMatchingOrder(mo);

  // allocate GPU worklist
  int2* d_wl;
  cudaMalloc((void**)&d_wl, std::max(sizeof(int2) * g.edges, sizeof(int) * g.nodes));
  CheckCuda(__LINE__);

  // run timed code section
  CPUTimer timer;
  timer.start();
  const ull count = occurrences(mo, d_g, d_wl, SMs, mTpSM);
  const double runtime = timer.elapsed();
  printf("runtime: %.6f s\n", runtime);
  printf("%lld occurrences\n\n", count);

  // clean up
  freeECLgraph(g);
  cudaFree(d_g.nindex);
  cudaFree(d_g.nlist);
  cudaFree(d_wl);
  CheckCuda(__LINE__);
  return 0;
}


