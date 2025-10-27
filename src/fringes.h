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

#ifndef ECL_FRIGES
#define ECL_FRIGES


static const int MaxCoreSize = 6;  // maximum number of nodes in core (must be <= 8)

enum {undecided, core, fringe};

using byte = unsigned char;

struct MatchingOrder {
  byte nodes;
  byte anchorsets;  // number of anchor sets
  byte startdeg;
  byte destdeg [MaxCoreSize - 1];
  byte list [MaxCoreSize - 1];  // bmp: 0 means cannot be, 1 means must be adjacent to
  byte anch [(1 << MaxCoreSize) - 1];  // anchors
  byte fcnt [(1 << MaxCoreSize) - 1];  // fringe counts
};


static void printMatchingOrder(const MatchingOrder mo)
{
  printf("starting from %c >= %d\n\n", 'A', mo.startdeg);
  for (int i = 0; i < mo.nodes - 1; i++) {
    printf("find %c >= %d\n", 'A' + i + 1, mo.destdeg[i]);
    printf(" connecting to");
    for (int j = 0; j <= i; j++) {
      if (mo.list[i] & (1 << j)) printf(" %c", 'A' + j);
    }
    if (mo.list[i] != ((2 << i) - 1)) {
      printf("\n skipping over");
      for (int j = 0; j <= i; j++) {
        if (~mo.list[i] & (1 << j)) printf(" %c", 'A' + j);
      }
    }
    printf("\n\n");
  }

  byte anchors = 0;
  for (int i = 0; i < mo.anchorsets; i++) {
    anchors |= mo.anch[i];
  }
  printf("anchors:");
  for (int i = 0; i < mo.nodes; i++) {
    if (anchors & (1 << i)) printf(" %c", 'A' + i);
  }
  printf("\n\n");

  printf("anchor sets\n");
  for (int i = 0; i < mo.anchorsets; i++) {
    int bmp = mo.anch[i];
    for (int j = 0; j < MaxCoreSize; j++) {
      if (bmp & (1 << j)) printf(" %c", 'A' + j);
    }
    printf(": %d\n", mo.fcnt[i]);
  }
  printf("\n");
}


#endif


