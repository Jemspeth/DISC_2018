import graph500 as g500
import numpy as np
import time
import sys

def main():

  if len(sys.argv) != 4 or int(sys.argv[3]) < 3 or int(sys.argv[2]) < 1 or 
    int(sys.argv[1]) < 1:
    exit_message = 'Input parameters incompatible with Generator. \
              \nUsage: python test.py <scale_left> <scale_right> <edge_factor> \
              \n\n<scale_left> : Number of vertices in left set (N) = 2^scale_left \
              \n<scale_right> : Number of vertices in right set (P) = 2^scale_right \
              \n<edge_factor> : Number of edges = N * edge_factor'
    exit(exit_message)

  scale_l = int(sys.argv[1])
  scale_r = int(sys.argv[2])
  edge_factor = int(sys.argv[3])

  print '------------------------- BIPARTITE GRAPH (NSKG) -------------------------'
  print 'Generating bipartite noisy stochastic kronecker graph...'
  start = time.time()
  edge_list = g500.bipartite_noisy_kronecker(scale_l, scale_r, edge_factor)
  end = time.time()
  print 'Total time to generate graph/s:', end - start, '\n'

  print 'Converting to adjacency list...'
  start = time.time()
  adj_list = g500.to_adj_list( edge_list )
  end = time.time()
  print 'Total time to convert to adjacency list:', end - start, '\n'

  print 'Graph statistics:'
  #g500.print_stats( edge_list )
  count_O, count_I = g500.bi_degree( edge_list[0:2, :] )
  edges = g500.count_edges( adj_list )
  left = count_O.size
  right = count_I.size
  ave_degree = 2*edges / float(left + right)
  left_degree = edges / float(left)
  right_degree = edges / float(right)
  print 'Average degree of entire graph', ave_degree
  print 'Average degree of left: ', left_degree
  print 'Average degree of right:', right_degree
  print 'Number of vertices in left:', left
  print 'Number of vertices in right:', right
  print 'Number of edges:', edges, '\n'
 
  edge_file = 'edge_%dx%dx%d.txt' % (scale_l, scale_r, edge_factor) 
  adj_file = 'adj_%dx%dx%d.txt' % (scale_l, scale_r, edge_factor)
  g500.write_edge_list( edge_list, left, right, edges, edge_file )
  g500.write_adj_list( adj_list, left, right, edges, adj_file )
  
  #print 'Starting Bipartite Matching in Python...'
  start = time.time()
  match = g500.bipartite_match( adj_list )
  end = time.time()
  total = end - start
  print 'Total time to complete matching: %f' % total
  print 'Total number of matches:', len(match[0])
  #g500.printMatching(match[0])
  
  g500.write_timings( left, right, edges, total, 'HopKarpTimes.txt' )
  
  g500.plot_points(count_O, 'Degree Distribution of Left Set')
  g500.plot_points(count_I, 'Degree Distribution of Right Set')
  #timings = g500.collect_timings()
  #g500.plot_scaling(timings)

if __name__ == '__main__':
	main()
