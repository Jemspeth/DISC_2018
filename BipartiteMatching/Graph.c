#include "Graph.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>


/* create a new graph with n vertices labeled 0..n-1 and no edges */
Graph* graph_create( char *filename )
{
  Graph* g;
  g = malloc(sizeof(Graph));
  g->l = 0;
  g->r = 0;
  g->m = 0;
  graph_read_adj_list( g, filename );
  return g;
}

/* free all space used by graph */
void graph_destroy( Graph *g )
{
  for(int i = 0; i < g->l; i++)
  {
    vec_free( vec_get(&g->alist, i) );
  }
  vec_free(&g->lverts);
  vec_free(&g->alist);
  free(g);
}


void graph_add_rverts( Graph *g, Vector *v )
{
  vec_add(&g->alist, v);
}


void graph_add_lverts( Graph *g, Vector *v )
{
  g->lverts = *v;
}


void graph_add_lvert( Graph *g, void *item )
{
  vec_add(&g->lverts, item);
}


void graph_read_adj_list( Graph *graph, char *filename )
{
  FILE *fp;
  fp = fopen(filename, "r");

  int *tmp;
  Vector *rv;
  char ch, buff[15], read[64];
  memset(buff, 0, sizeof(buff));

  fgets(read, sizeof(read), fp);
  sscanf(read, "%d,%d,%d", &graph->l, &graph->r, &graph->m);

  vec_init(&graph->lverts, graph->l);
  vec_init(&graph->alist, graph->r);

  while(1)
  {
    ch = fgetc(fp);
    if( ch == EOF )
      break;
    else if( (ch >= '0' && ch <= '9') || ch == '-' )
    {
      str_app(buff, ch);
    }
    else if( ch == ',' )
    {
      tmp = malloc(sizeof(int));
      *tmp = atoi(buff);
      vec_add(rv, tmp);
      memset(buff, 0, sizeof(buff));
    }
    else if ( ch == '[' )
    {
      tmp = malloc(sizeof(int));
      *tmp = atoi(buff);
      graph_add_lvert(graph, tmp);
      rv = malloc(sizeof(Vector));
      vec_init(rv, 1);
      memset(buff, 0, sizeof(buff));
    }
    else if( ch == ']' )
    {
      tmp = malloc(sizeof(int));
      *tmp = atoi(buff);
      vec_add(rv, tmp);
      graph_add_rverts(graph, rv);
      memset(buff, 0, sizeof(buff));
      fgetc(fp);
    }
  }

  fclose(fp);
}


void graph_print( Graph *g )
{
  printf("\nNumber of Vertices in left: %d\n", g->l);
  printf("Number of Vertices in right: %d\n", g->r);
  printf("Number of edges: %d\n", g->m);
  printf("\nLeft Vertices: \n");
  vec_print(&g->lverts);
  printf("\nRight Vertices: \n");
  for( int i = 0; i < g->l; i++ )
  {
    vec_print( vec_get(&g->alist, i) );
  }
}


void str_app( char *str, char ch )
{
  int len = strlen(str);
  str[len] = ch;
  str[len + 1] = '\0';
}