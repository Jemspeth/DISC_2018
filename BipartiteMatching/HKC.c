#include <stdio.h>
#include "HopKarp.h"

int main( int argc, char **argv )
{
	Graph *graph = graph_create( argv[1] );
	//graph_print(graph);
  int match_count = hop_karp(graph);
  printf("Total number of matches: %d\n", match_count);
	graph_destroy( graph );
}