import graph500 as g500
import numpy as np
import time
import sys

SCALE_1 = int(sys.argv[1])
SCALE_2 = int(sys.argv[2])
EDGEFACTOR = int(sys.argv[3])

'''
SCALE_1 = 18
SCALE_2 = 18
EDGEFACTOR = 8
'''

def main():
  print '------------------------- BIPARTITE GRAPH (NSKG) -------------------------'
  print 'Generating bipartite noisy stochastic kronecker graph...'
  start = time.time()
  biG = g500.bipartite_noisy_kronecker(SCALE_1, SCALE_2, EDGEFACTOR)
  end = time.time()
  print 'Total time to generate graph/s:', end - start, '\n'

  print 'Converting to adjacency list...'
  start = time.time()
  adjList = g500.toAdjList( biG )
  end = time.time()
  print 'Total time to convert to adjacency list:', end - start, '\n'

  print 'Graph statistics:'
  count_O, count_I = g500.biDegree( biG[0:2, :] )
  edges = g500.count_edges( adjList )
  left = count_O.size
  right = count_I.size
  ave_degree = 2*edges / float(left + right)
  print 'Number of Vertices in first set:', left
  print 'Number of Vertices in second set:', right
  print 'Number of edges:', edges
  print 'Average degree: ', ave_degree, '\n'

	
  print 'Starting Hopcroft-Karp maximum bipartite matching...'
  start = time.time()
  match = g500.bipartiteMatch( adjList )
  end = time.time()
  total = end - start
  print 'Total time to find maximum bipartite matching:', total
	
  with open('./HopKarpTimes.txt', 'a') as file:
  	file.write('V1: ' + str(left) + '\n')
  	file.write('V2: ' + str(right) + '\n')
  	file.write('Edges: ' + str(edges) + '\n')
  	file.write('Total time: ' + str(total) + '\n\n')

  g500.writeAdjList( adjList, left, right, edges, 'output.txt')
  #timings = g500.collect_timings()
  #g500.plotScaling(timings)

if __name__ == '__main__':
	main()
