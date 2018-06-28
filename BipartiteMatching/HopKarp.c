#include "HopKarp.h"
#include "Vector.h"
#include "Graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>


int hop_karp( Graph *graph )
{
  // pairU[u] stores pair of u in matching where u
  // is a vertex on left side of Bipartite Graph.
  // If u doesn't have any pair, then pairU[u] is NIL
  //void **pairU = calloc(graph->l + 1, sizeof(void*));
  int *pairU = calloc(graph->l + 1, sizeof(int));

  // pairV[v] stores pair of v in matching. If v
  // doesn't have any pair, then pairU[v] is NIL
  //void **pairV = calloc(graph->r + 1, sizeof(void*));
  int *pairV = calloc(graph->r+1, sizeof(int));

  // dist[u] stores distance of left side vertices
  // dist[u] is one more than dist[u'] if u is next
  // to u'in augmenting path
  int *dist = malloc(sizeof(int) * (graph->l + 1));

  int result = 0;

  while( bfs(graph, dist, pairU, pairV) )
  {
    for( int i = 1; i <= graph->l; i++ )
    {
      printf("%d\n", i);
      if( pairU[i] == 0 && dfs(i, graph, dist, pairU, pairV) )
        result++;
    }
  }

  free(pairU);
  free(pairV);
  free(dist);
  
  return result;
}


int bfs( Graph *graph, int *dist, int *pairU, int *pairV )
{
	Vector Q;
	vec_init(&Q, 1);

  for( int i = 1; i <= graph->l; i++ )
  {
    if( pairU[i] == 0 )
    {
      dist[i] = 0;
      vec_add_int(&Q, i);
    }
    else
      dist[i] = INT_MAX;
  }

  dist[0] = INT_MAX;

  while( !vec_empty(&Q) )
  {
    int u = *(int*)vec_front(&Q);
    vec_qpop(&Q);

    Vector *tmp = vec_get(&graph->alist, u - 1);

    if( dist[u] < dist[0] )
    {
      for( int i = 0; i < vec_total(tmp); i++ )
      {
        int v = *(int*)vec_get(tmp, i);

        if( dist[pairV[v]] == INT_MAX )
        {
          dist[pairV[v]] = dist[u] + 1;
          vec_add_int(&Q, pairV[v]);
        }
      }
    }
  }

  vec_free(&Q);
  return dist[0] != INT_MAX;
}

int dfs( int u, Graph *graph, int *dist, int *pairU, int *pairV )
{
  if( u != 0 )
  {
    Vector *tmp = vec_get(&graph->alist, u - 1);
    vec_print(tmp);

    for( int i = 0; i < vec_total(tmp); i++ )
    {
      int v = *(int*)vec_get(tmp, i);
      if( dist[pairV[v]] == dist[u] + 1 )
      {
        if( dfs( pairV[v], graph, dist, pairU, pairV ) == 1 )
        {
          pairV[v] = u;
          pairU[u] = v;
          return 1;
        }
      }
    }

    dist[u] = INT_MAX;
    return 0;
  }
  return 1;
}

/*
int bfs( Graph *graph, void **dist, void **pairU, void **pairV )
{
  Vector Q;
  vec_init(&Q, 1);

  for( int i = 1; i <= graph->m; i++ )
  {
    if( *(int*)pairU[i] == 0 )
    {
      *(int*)dist[i] = 0;
      vec_add_int(&Q, i);
    }
    else
      *(int*)dist[i] = INT_MAX;
  }

  *(int*)dist[0] = INT_MAX;

  while( !vec_empty(&Q) )
  {
    int u = *(int*)vec_front(&Q);
    vec_qpop(&Q);

    Vector *tmp = vec_get(&graph->alist, u);

    if( *(int*)dist[u] < *(int*)dist[0] )
    {
      for( int i = 0; i < vec_total(tmp); i++ )
      {
        int v = *(int*)vec_get(tmp, i);

        if( *(int*)dist[*(int*)pairV[v]] == INT_MAX )
        {
          *(int*)dist[*(int*)pairV[v]] = *(int*)dist[u] + 1;
          int *pv = malloc(sizeof(int));
          *pv = *(int*)pairV[v];
          vec_add(&Q, pv);
        }
      }
    }
  }

  vec_free(&Q);
  return *(int*)dist[0] != INT_MAX;
}*/

/*
int dfs( int vert, Graph *graph, void **dist, void **pairU, void **pairV )
{
  if( vert != 0 )
  {
    Vector *tmp = vec_get(&graph->alist, vert);
    for( int i = 0; i < vec_total(tmp); i++ )
    {
      int v = *(int*)vec_get(tmp, i);
      if( *(int*)dist[*(int*)pairV[v]] == *(int*)dist[vert] + 1 )
      {
        if( dfs(*(int*)dist[*(int*)pairV[v]], graph, dist, pairU, pairV) == 1 )
        {
          *(int*)pairV[v] = vert;
          *(int*)pairU[vert] = v;
          return 1;
        }
      }
    }

    *(int*)dist[vert] = INT_MAX;
    return 0;
  }
  return 1;
}*/