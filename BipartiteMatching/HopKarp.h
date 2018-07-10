#ifndef HK_H
#define HK_H

#include "Graph.h"

int hop_karp( Graph *graph, int **Uverts, int **Vverts );
int bfs( Graph *graph, int *dist, int *pairU, int *pairV );
int dfs( int vert, Graph *graph, int *dist, int *pairU, int *pairV );
void print_matching( Graph *graph, int *Uverts, int *Vverts );
void mem_testing( Graph *graph );

#endif