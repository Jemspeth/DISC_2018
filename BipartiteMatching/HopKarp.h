#ifndef HK_H
#define HK_H

#include "Graph.h"

int hop_karp( Graph *graph );
int bfs( Graph *graph, int *dist, int *pairU, int *pairV );
int dfs( int vert, Graph *graph, int *dist, int *pairU, int *pairV );
//int bfs( Graph *graph, void **dist, void **pairU, void **pairV );
//int dfs( int vert, Graph *graph, void **dist, void **pairU, void **pairV );

#endif