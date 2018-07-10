#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "HopKarp.h"

int main( int argc, char **argv )
{
  int *Uverts, *Vverts;
  clock_t start, end;
  double cpu_time_used;

  Graph *graph = graph_create( argv[1] );
  printf("Done Reading Graph\n");
  graph_print( graph );

  //printf("\nStarting Bipartite Matching in C...\n");
  start = clock();
  int match_count = hop_karp( graph, &Uverts, &Vverts );
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Total time to complete matching: %f\n", cpu_time_used);

  printf("Total number of matches: %d\n", match_count);
  //print_matching( graph, Uverts, Vverts );

  free(Uverts);
  free(Vverts);
	graph_destroy( graph );
}