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
#include <queue>
#include <algorithm>
#include <cassert>
#include "patterns.h"
#include "fringes.h"


static MatchingOrder genCoreMatchingOrder(const int src, const pattern_t motif, const char state [MaxPatternSize])
{
  assert(state[src] == core);
  const int num = motif.nodes;

  int map [MaxPatternSize];
  map[src] = 0;

  bool visited [MaxPatternSize];
  for (int v = 0; v < num; v++) visited[v] = false;
  visited[src] = true;

  MatchingOrder mo;
  mo.startdeg = motif.adj[src].size();

  int top = 0;
  while (top < num - 1) {
    // pick destination with most constraints
    int dst = -1;
    int con = -1;
    for (int v = 0; v < num; v++) {
      if ((!visited[v]) && (state[v] == core)) {
        // count constraints
        int cntu = 0;
        for (int u = 0; u < num; u++) {
          if (visited[u] && (motif.adj[v].find(u) != motif.adj[v].end())) cntu++;
        }

        // pick node with most constraints (or highest degree if tie (or lowest first connection if still a tie))
        if ((con < cntu) || ((con == cntu) && (motif.adj[dst].size() < motif.adj[v].size())) || ((con == cntu) && (motif.adj[dst].size() == motif.adj[v].size()) && (*motif.adj[dst].begin() > *motif.adj[v].begin()))) {
          dst = v;
          con = cntu;
        }
      }
    }

    if (dst < 0) break;

    map[dst] = top + 1;
    mo.destdeg[top] = motif.adj[dst].size();

    // populate list of connections
    mo.list[top] = 0;
    for (int v = 0; v < num; v++) {
      if (visited[v] && (motif.adj[dst].find(v) != motif.adj[dst].end())) {
        mo.list[top] |= 1 << map[v];
      }
    }

    visited[dst] = true;
    top++;
  }
  mo.nodes = top + 1;

  // figure out anchor sets and counts
  int as [1 << MaxCoreSize];
  for (int i = 0; i < (1 << MaxCoreSize); i++) as[i] = 0;

  for (int v = 0; v < num; v++) {
    if (state[v] == fringe) {
      int bmp = 0;
      for (auto i: motif.adj[v]) {
        bmp |= 1 << map[i];
      }
      assert(bmp != 0);
      as[bmp]++;
    }
  }

  mo.anchorsets = 0;
  for (int i = 1; i < (1 << MaxCoreSize); i++) {
    if (as[i] > 0) {
      mo.anch[mo.anchorsets] = i;
      mo.fcnt[mo.anchorsets] = as[i];
      mo.anchorsets++;
    }
  }

  return mo;
}


static void outputMotif(const char* const filename, const pattern_t motif, const char* const state = NULL)
{
  // create dot file
  FILE* f = fopen(filename, "w");
  fprintf(f, "graph {\n");
  if (state != NULL) {
    for (int i = 0; i < motif.nodes; i++) {
      if (state[i] == core) {
        fprintf(f, "  %d [style=filled, fillcolor=\"#9090ff\", label=\"%d\"]\n", i, i);  // blue
      } else {
        fprintf(f, "  %d [style=filled, fillcolor=\"#ffffff\", label=\"%d\"]\n", i, i);  // white
      }
    }
  }
  for (int i = 0; i < motif.nodes; i++) {
    for (auto j: motif.adj[i]) {
      if (i < j) {
        fprintf(f, "  %d -- %d\n", i, j);
      }
    }
  }
  fprintf(f, "}\n");
  fclose(f);
}


static bool isConnected(const pattern_t motif)
{
  // init
  bool reached [MaxPatternSize];
  for (int i = 0; i < motif.nodes; i++) {
    reached[i] = false;
  }

  // BFS
  std::queue<int> q;
  q.push(0);
  do {
    const int u = q.front();
    q.pop();
    if (!reached[u]) {
      reached[u] = true;
      for (auto v: motif.adj[u]) {
        if (!reached[v]) {
          q.push(v);
        }
      }
    }
  } while (!q.empty());

  // check if all nodes reached
  for (int i = 0; i < motif.nodes; i++) {
    if (!reached[i]) return false;
  }
  return true;
}


static void connectCore(const pattern_t motif, char state [MaxPatternSize])
{
  bool reached [MaxPatternSize];
  bool visited [MaxPatternSize];

  do {
    // init
    int c = -1;
    for (int u = 0; u < motif.nodes; u++) {
      reached[u] = true;
      if (state[u] == core) {
        reached[u] = false;
        c = u;
      }
    }
    assert(c >= 0);

    // BFS
    std::queue<int> q;
    q.push(c);
    do {
      const int u = q.front();
      q.pop();
      if (!reached[u]) {
        reached[u] = true;
        for (auto v: motif.adj[u]) {
          if (!reached[v]) {
            q.push(v);
          }
        }
      }
    } while (!q.empty());

    // check if all nodes reached
    int u = 0;
    while ((u < motif.nodes) && (reached[u])) {
      u++;
    }
    if (u >= motif.nodes) break;  // all connected

    // init Q with reachable core nodes
    for (int u = 0; u < motif.nodes; u++) {
      visited[u] = false;
      if ((state[u] == core) && (reached[u])) {
        visited[u] = true;
        q.push(u);
      }
    }

    // do BFS thru unvisited nodes to nearest unvisited core node
    do {
      const int u = q.front();
      q.pop();
      for (auto v: motif.adj[u]) {
        if (!visited[v]) {
          visited[v] = true;
          if (!reached[v]) {
            printf("including %d in core vertices\n", u);
            state[u] = core;  // add to core vertices
            while (!q.empty()) q.pop();  // drain Q
            break;  // stop
          } else {
            q.push(v);
          }
        }
      }
    } while (!q.empty());
  } while (true);
}


static void determineCore(const pattern_t motif, char state [MaxPatternSize])
{
  // init
  for (int i = 0; i < motif.nodes; i++) {
    state[i] = undecided;
  }

  // greedily turn undecided deg-X nodes into fringes
  for (int x = 1; x < motif.nodes; x++) {
    for (int u = motif.nodes - 1; u >= 0; u--) {
      if (state[u] == undecided) {
        if (motif.adj[u].size() == x) {
          state[u] = fringe;
          for (auto v: motif.adj[u]) {
            assert(state[v] != fringe);
            state[v] = core;
          }
        }
      }
    }
  }

  // connect core if necessary
  connectCore(motif, state);
}


int main(int argc, char* argv [])
{
  printf("Fringe Preprocessor v0.06 (%s)\n", __FILE__);
  printf("Copyright 2025 Texas State University\n\n");

  if ((argc < 2) || ((argc % 2) != 0)) {
    printf("USAGE: %s motif_number\n", argv[0]);
    printf("USAGE: %s number_of_nodes edge1_source edge1_destination ...\n\n", argv[0]);
    return -1;
  }

  pattern_t motif;
  if (argc == 2) {
    // predefined motif
    const int number = atoi(argv[1]);
    motif = getPattern(number);
  } else {
    // read motif from command line
    motif.name = "user defined";
    const int n = atoi(argv[1]);
    if ((n < 2) || (n > MaxPatternSize)) {fprintf(stderr, "ERROR: number_of_nodes must be between 2 and %d\n", MaxPatternSize); exit(-1);}
    motif.nodes = n;
    for (int p = 2; p + 1 < argc; p += 2) {
      const int s = atoi(argv[p]);
      const int d = atoi(argv[p + 1]);
      if ((s < 0) || (s >= n)) {fprintf(stderr, "ERROR: edge source must be between 0 and %d\n", n - 1); exit(-1);}
      if ((d < 0) || (d >= n)) {fprintf(stderr, "ERROR: edge destination must be between 0 and %d\n", n - 1); exit(-1);}
      motif.adj[s].insert(d);
      motif.adj[d].insert(s);
    }
  }

  // print motif info
  printf("motif: %s\n", motif.name.c_str());
  int m = 0;
  for (int i = 0; i < motif.nodes; i++) {
    m += motif.adj[i].size();
  }
  printf("%d nodes and %d edges\n", motif.nodes, m / 2);

  if (!isConnected(motif)) {fprintf(stderr, "ERROR: motif is disconnected\n"); exit(-1);}

  char state [MaxPatternSize];
  determineCore(motif, state);

  outputMotif("motif.dot", motif, state);

  int c = 0;
  for (int i = 0; i < motif.nodes; i++) {
    if (state[i] == core) c++;
  }
  printf("%d core nodes\n\n", c);

  if (c > MaxCoreSize) {fprintf(stderr, "ERROR: only motifs with up to %d core nodes supported\n", MaxCoreSize); exit(-1);}

  // find highest degree core vertex
  int hisrc, hideg = -1;
  for (int i = 0; i < motif.nodes; i++) {
    if (state[i] == core) {
      const int deg = motif.adj[i].size();
      if (hideg < deg) {
        hideg = deg;
        hisrc = i;
      }
    }
  }

  const MatchingOrder mo = genCoreMatchingOrder(hisrc, motif, state);
  printMatchingOrder(mo);

  FILE* f = fopen("motif.mo", "wb");
  const int size = fwrite(&mo, 1, sizeof(MatchingOrder), f);
  fclose(f);
  if (size != sizeof(MatchingOrder)) {fprintf(stderr, "ERROR: could not write matching order to file\n"); exit(-1);}

  printf("wrote matching order to file 'motif.mo'\n\n");

  return 0;
}


/*
  dot -Tpng motif.dot > motif.png
*/


