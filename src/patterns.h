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
#ifndef ECL_PATTERNS
#define ECL_PATTERNS

#include <string>
#include <set>

static const int MaxPatternSize = 32;  // maximum number of nodes in pattern

enum {A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P};

using set = std::set<signed char>;

struct pattern_t {
  std::string name;
  signed char nodes;
  set adj [MaxPatternSize];
};

static pattern_t getPattern(const int pattern)
{
  switch (pattern) {
    case 0:
      return {"user-defined pattern",
        16,
        set{B, C, D, E, F, G, O, P},
        set{A, C, F, G, H, L, M, N, O, P},
        set{A, B, I, J, K, L, M, N, O, P},
        set{A, E},//
        set{A, D},//
        set{A, B},
        set{A, B},
        set{B},
        set{C},
        set{C},
        set{C},
        set{B, C},
        set{B, C},
        set{B, C},
        set{A, B, C},
        set{A, B, C}
      };
    /************************* all 1-vertex motifs *************************/
/*
    case 0:
      return {"1-vertex pattern 0 [single vertex]",
        1,
        set{}
      };
*/
    /************************* all 2-vertex motifs *************************/
    case 1:
      return {"2-vertex pattern 1 (1 tails) [single edge]: vertex core",
        2,
        set{B},
        set{A}
      };
    /************************* all 3-vertex motifs *************************/
    case 2:
      return {"3-vertex pattern 3 (2 tails) [wedge]: vertex core",
        3,
        set{B, C},
        set{A},
        set{A}
      };
    case 3:
      return {"3-vertex pattern 7 [triangle]: edge core",
        3,
        set{B, C},
        set{A, C},
        set{A, B}
      };
    /************************* all 4-vertex motifs *************************/
    case 4:
      return {"4-vertex pattern 7 (3 tails) [3-star]: vertex core",
        4,
        set{B, C, D},
        set{A},
        set{A},
        set{A}
      };
    case 5:
      return {"4-vertex pattern 13 (1+1 tails) [4-path]: edge core",
        4,
        set{B, D},
        set{A, C},
        set{B},
        set{A}
      };
    case 6:
      return {"4-vertex pattern 15 (1 tails) [tailed triangle]: edge core",
        4,
        set{B, C, D},
        set{A, C},
        set{A, B},
        set{A}
      };
    case 7:
      return {"4-vertex pattern 30 [4-cycle]: wedge core",
        4,
        set{C, D},
        set{C, D},
        set{A, B},
        set{A, B}
      };
    case 8:
      return {"4-vertex pattern 31 [diamond]: edge core",
        4,
        set{B, C, D},
        set{A, C, D},
        set{A, B},
        set{A, B}
      };
    case 9:
      return {"4-vertex pattern 63 [4-clique]: triangle core",
        4,
        set{B, C, D},
        set{A, C, D},
        set{A, B, D},
        set{A, B, C}
      };
    /************************* all 5-vertex motifs *************************/
    case 10:
      return {"5-vertex pattern 15 (4 tails): vertex core",
        5,
        set{B, C, D, E},
        set{A},
        set{A},
        set{A},
        set{A}
      };
    case 11:
      return {"5-vertex pattern 29 (2+1 tails): edge core",
        5,
        set{B, D, E},
        set{A, C},
        set{B},
        set{A},
        set{A}
      };
    case 12:
      return {"5-vertex pattern 31 (2 tails) [2-tailed triangle]: edge core",
        5,
        set{B, C, D, E},
        set{A, C},
        set{A, B},
        set{A},
        set{A}
      };
    case 13:
      return {"5-vertex pattern 58 (1+1 tails) [5-path]: wedge core",
        5,
        set{C, E},
        set{C, D},
        set{A, B},
        set{B},
        set{A}
      };
    case 14:
      return {"5-vertex pattern 59 (1+1 tails) [1+1-tailed triangle]: edge core",
        5,
        set{B, C, E},
        set{A, C, D},
        set{A, B},
        set{B},
        set{A}
      };
    case 15:
      return {"5-vertex pattern 62 (1 tails) [tailed 4-cycle]: wedge core",
        5,
        set{C, D, E},
        set{C, D},
        set{A, B},
        set{A, B},
        set{A}
      };
    case 16:
      return {"5-vertex pattern 63 (1 tails) [tailed diamond]: edge core",
        5,
        set{B, C, D, E},
        set{A, C, D},
        set{A, B},
        set{A, B},
        set{A}
      };
    case 17:
      return {"5-vertex pattern 126 [4-cycle with center element on diagonal]: wedge core",
        5,
        set{C, D, E},
        set{C, D, E},
        set{A, B},
        set{A, B},
        set{A, B}
      };
    case 18:
      return {"5-vertex pattern 127: edge core",
        5,
        set{B, C, D, E},
        set{A, C, D, E},
        set{A, B},
        set{A, B},
        set{A, B}
      };
    case 19:
      return {"5-vertex pattern 185 (1 tails) [triangle with double long tail]: wedge core",
        5,
        set{B, E},
        set{A, C, D},
        set{B, D},
        set{B, C},
        set{A}
      };
    case 20:
      return {"5-vertex pattern 187 (1 tails) [tailed diamond]: triangle core",
        5,
        set{B, C, E},
        set{A, C, D},
        set{A, B, D},
        set{B, C},
        set{A}
      };
    case 21:
      return {"5-vertex pattern 191 (1 tails) [tailed 4-clique]: triangle core",
        5,
        set{B, C, D, E},
        set{A, C, D},
        set{A, B, D},
        set{A, B, C},
        set{A}
      };
    case 22:
      return {"5-vertex pattern 207 [hour glass]: wedge core",
        5,
        set{B, C, D, E},
        set{A, E},
        set{A, D},
        set{A, C},
        set{A, B}
      };
    case 23:
      return {"5-vertex pattern 220 [5-cycle]",
        5,
        set{D, E},
        set{C, E},
        set{B, D},
        set{A, C},
        set{A, B}
      };
    case 24:
      return {"5-vertex pattern 221 [house]: wedge core",
        5,
        set{B, D, E},
        set{A, C, E},
        set{B, D},
        set{A, C},
        set{A, B}
      };
    case 25:
      return {"5-vertex pattern 223 [triangle with 2 wedge fringes]: triangle core",
        5,
        set{B, C, D, E},
        set{A, C, E},
        set{A, B, D},
        set{A, C},
        set{A, B}
      };
    case 26:
      return {"5-vertex pattern 254: wedge core",
        5,
        set{C, D, E},
        set{C, D, E},
        set{A, B, D},
        set{A, B, C},
        set{A, B}
      };
    case 27:
      return {"5-vertex pattern 255",
        5,
        set{B, C, D, E},
        set{A, C, D, E},
        set{A, B, D},
        set{A, B, C},
        set{A, B}
      };
    case 28:
      return {"5-vertex pattern 495 [4-cycle with A in middle]",
        5,
        set{B, C, D, E},
        set{A, D, E},
        set{A, D, E},
        set{A, B, C},
        set{A, B, C}
      };
    case 29:
      return {"5-vertex pattern 511 [5-clique with 1 missing edge]",
        5,
        set{B, C, D, E},
        set{A, C, D, E},
        set{A, B, D, E},
        set{A, B, C},
        set{A, B, C}
      };
    case 30:
      return {"5-vertex pattern 1023 [5-clique]",
        5,
        set{B, C, D, E},
        set{A, C, D, E},
        set{A, B, D, E},
        set{A, B, C, E},
        set{A, B, C, D}
      };
    default: {
      printf("ERROR: unknown pattern\n\n");
      exit(-1);
    }
  }
}

#endif


