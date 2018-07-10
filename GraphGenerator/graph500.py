import numpy as np
import random
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import math

# Generates a graph with number of vertices as 2^scale, respectively. The total
# number of edges will be approx. edge_factor * 2^scale. 
# Returns: edge list as numpy array of dimensions 3 x edgeCount
def kronecker_generator( scale, edge_factor ):
	N = 2 ** scale
	M = edge_factor * N

  #A, B, C, and D determine the shape of the degree distribution.
  #noise determines the amount of smoothing to be applied to the distribution
	A = 0.57 #0.57
	B = 0.20 #0.10
	C = 0.20 #0.19
	D = 0.05
	noise = 0.05

	ijw = np.ones((3, M))

	for i in range(scale):
		ab = A + B
		c_norm = C/(1 - (A + B))
		a_norm = A/(A + B)
		u = random.uniform( -noise, noise )
		A = A - ((2 * u * A)/(A + D))
		B = B + u
		C = C + u
		D = D - ((2 * u * D)/(A + D))

		i_bit = np.random.rand(1, M) > ab
		j_bit = np.greater( np.random.rand(1, M), ((c_norm * i_bit) + (a_norm * ~i_bit)) )
		ijw[0, :] = ijw[0, :] + (2**i * i_bit)
		ijw[1, :] = ijw[1, :] + (2**i * j_bit)

	ijw[0:2, :] = ijw[0:2, :] - 1

	ijw[2, :] = np.random.rand(1, M)

	p = np.random.permutation(N)
	ijw[0:2, :] = p[ijw[0:2, :].astype(int)]

	p = np.random.permutation(M)
	ijw = ijw[:, p]

	return ijw


# Generates a bipartite graph with number of vertices in left and right set
# determined by parameters scale_l and scale_r. The total number of edges will be
# approx. edge_factor * 2^scale_l. The vertices in the left set are negative to
# easily differentiate the two sets.
#
# scale_l : number of vertices in left, N, where N = 2^scale_l
# scale_r : number of vertices in right, P, where P = 2^scale_r
# edge_factor : number of edges, M, where M = edge_factor * N
# Returns: edge list as numpy array of dimensions 3 x edgeCount
def bipartite_noisy_kronecker( scale_l, scale_r, edge_factor ):
	N = 2 ** scale_l
	P = 2 ** scale_r
	M = ((edge_factor-2) * N) / 2
	noise = 0.05

  #These determine the shape of the degree distribution for left set.
  #noise determines the amount of smoothing to be applied to the distribution
  #For a somewhat realistic baseline use [0.57, 0.19, 0.19, 0.05]
	A_I = 0.57
	B_I = 0.19
	C_I = 0.19
	D_I = 0.05

  #These determine the shape of the degree distribution for right set.
  #noise determines the amount of smoothing to be applied to the distribution
	A_O = 0.57
	B_O = 0.19
	C_O = 0.19
	D_O = 0.05

	i = np.zeros(M)
	j = np.zeros(M)

  # Generate vertices present in left set
	for c in range(scale_l):
		ab_I = A_I + B_I
		c_norm_I = C_I/(1 - (A_I + B_I))
		a_norm_I = A_I/(A_I + B_I)

    #Apply noise at each level to remove oscillations
		u = random.uniform( -noise, noise )
		A_I = A_I - ((2 * u * A_I)/(A_I + D_I))
		B_I = B_I + u
		C_I = C_I + u
		D_I = D_I - ((2 * u * D_I)/(A_I + D_I))

		i_bit = np.random.rand(1, M) > ab_I
		j_bit = np.greater( np.random.rand(1, M), ((c_norm_I * i_bit) + (a_norm_I * ~i_bit)) )
		i = i + 2**(c) * i_bit
		j = j + 2**(c) * j_bit

	ij = np.append(i, j, axis=1)

	a = np.zeros(M)
	b = np.zeros(M)

  # Generate vertices present in right set
	for d in range(scale_r):
		ab_O = A_O + B_O
		c_norm_O = C_O/(1 - (A_O + B_O))
		a_norm_O = A_O/(A_O + B_O)

    #Apply noise at each level to remove oscillations
		u = random.uniform( -noise, noise )
		A_O = A_O - ((2 * u * A_O)/(A_O + D_O))
		B_O = B_O + u
		C_O = C_O + u
		D_O = D_O - ((2 * u * D_O)/(A_O + D_O))

		a_bit = np.random.rand(1, M) > ab_O
		b_bit = np.greater( np.random.rand(1, M), ((c_norm_O * a_bit) + (a_norm_O * ~a_bit)) )
		a = a + 2**(d) * a_bit
		b = b + 2**(d) * b_bit

	ab = np.append(a, b, axis=1)

	edge_list = np.ones( (3, 2*M) )
	edge_list[0, :] = ij
	edge_list[1, :] = ab
	edge_list[2, :] = np.random.rand(1, 2*M)

  # Make sure to cover all elements from 1 to 2^scale for both sets
	#edge_list = cover_sets( edge_list, N, P )

  # Permute the vertices and edges
	perm_N = np.random.permutation(N)
	edge_list[0, :] = -1 - perm_N[ edge_list[0, :].astype(int) ]
	perm_P = np.random.permutation(P)
	edge_list[1, :] = 1 + perm_P[ edge_list[1, :].astype(int) ]
	perm = np.random.permutation(edge_list[0].size)
	edge_list = edge_list[:, perm]
	
	return edge_list


# All vertices in the set from 1..(2^scale_x) should have at minimum one edge
# in the edgelist. This covers both the left and right sets without modifying
# the degree distribution much.
def cover_sets( edge_list, N, P ):
	left_cover = np.random.permutation(N).reshape((1, N))
	left_cover = np.append(left_cover, np.random.randint(1, P, (1, N)), axis=0)
	left_cover = np.append(left_cover, np.random.rand(1, N), axis=0)

	right_cover = np.random.randint(0, N, (1, P))
	all_right = np.random.permutation(P).reshape((1,P))
	right_cover = np.append(right_cover, all_right, axis=0)
	right_cover = np.append(right_cover, np.random.rand(1, P), axis=0)

	edge_list = np.append(edge_list, left_cover, axis=1)
	edge_list = np.append(edge_list, right_cover, axis=1)

	return edge_list


# Find maximum cardinality matching of a bipartite graph (U,V,E).
# The output is a triple (M,A,B) where M is a
# dictionary mapping members of V to their matches in U, A is the part
# of the maximum independent set in U, and B is the part of the MIS in V.
# The same object may occur in both U and V, and is treated as two
# distinct vertices if this happens.
# graph : dictionary mapping members of U to a list of their neighbors in V
def bipartite_match( graph ):
	# initialize greedy matching (redundant, but faster than full search)
	matching = {}
	for u in graph:
		for v in graph[u]:
			if v not in matching:
				matching[v] = u
				break
	
	while 1:
		# structure residual graph into layers
		# pred[u] gives the neighbor in the previous layer for u in U
		# preds[v] gives a list of neighbors in the previous layer for v in V
		# unmatched gives a list of unmatched vertices in final layer of V,
		# and is also used as a flag value for pred[u] when u is in the first layer
		preds = {}
		unmatched = []
		pred = dict([(u,unmatched) for u in graph])
		for v in matching:
			del pred[matching[v]]
		layer = list(pred)
		
		# repeatedly extend layering structure by another pair of layers
		while layer and not unmatched:
			newLayer = {}
			for u in layer:
				for v in graph[u]:
					if v not in preds:
						newLayer.setdefault(v,[]).append(u)
			layer = []
			for v in newLayer:
				preds[v] = newLayer[v]
				if v in matching:
					layer.append(matching[v])
					pred[matching[v]] = v
				else:
					unmatched.append(v)
					
		# did we finish layering without finding any alternating paths?
		if not unmatched:
			unlayered = {}
			for u in graph:
				for v in graph[u]:
					if v not in preds:
						unlayered[v] = None
			return (matching,list(pred),list(unlayered))

		# recursively search backward through layers to find alternating paths
		# recursion returns true if found path, false otherwise
		def recurse(v):
			if v in preds:
				L = preds[v]
				del preds[v]
				for u in L:
					if u in pred:
						pu = pred[u]
						del pred[u]
						if pu is unmatched or recurse(pu):
							matching[v] = u
							return 1
			return 0

		for v in unmatched: recurse( v )


# Returns adjacency representation of edgelist
# NOTE: By using sets, duplicate edges are eliminated from the graph
# edge_list : edge list to be converted to adjacency list
def to_adj_list( edge_list ):
	adj_list = defaultdict(set)
	for i in edge_list.T:
		adj_list[int(i[0])].add(int(i[1]))

	for i in adj_list:
		adj_list[i] = list(adj_list[i])

	return adj_list


# Writes adjacency list to a file in the form of:
# N, P, M
# -1 [24, 23]
# -32 [6]
# -31 [19, 6, 31]
# 
# adj_list : Adjacency list to be written
# left : number of vertices in the left set
# right : number of vertices in the right set
# edges : number of edges in the graph
# filename : name of the file to write
def write_adj_list( adj_list, left, right, edges, filename ):
	with open(filename, 'w') as outfile:
		outfile.write(str(left) + ',' + str(right) + ',' + str(edges) + '\n')
		for i in adj_list:
			outfile.write(str(i) + ' ' + str(adj_list[i]) + '\n')


# Writes edge list to a file in the form of:
# N, P, M
# -16 11
# -4 8
# -25 15
#
# edge_list : Edge list to be written
# left : number of vertices in the left set
# right : number of vertices in the right set
# edges : number of edges in the graph
# filename : name of the file to write
def write_edge_list( edge_list, left, right, edges, filename ):
	with open(filename, 'w') as outfile:
		outfile.write(str(left) + ',' + str(right) + ',' + str(edges) + '\n')
		for i in range(edges):
			outfile.write(str(int(edge_list[0][i])) + ' ' + str(int(edge_list[1][i])))
			outfile.write(' 1\n')


# Returns the total number of edges in an adjacency list
def count_edges( adj_list ):
	count = sum(len(v) for v in adj_list.itervalues())
	return count


# Returns an edgelist that has been made undirected
def make_undirected( ijw ):
	jiw = np.asarray([ijw[1,:], ijw[0,:], ijw[2,:]])
	ijw = np.append(ijw, jiw, axis=1)
	return ijw


# Removes self edges in an edge list
def remove_self( ijw ):
	ijw = ijw[:, ijw[0,:] != ijw[1,:]]
	return ijw


# Removes duplicate edges in an edge list
def remove_duplicate( ijw ):
	dt = np.dtype((np.void, ijw.dtype.itemsize * ijw.shape[0]))
	data = np.asfortranarray( ijw ).view(dt)
	unique_ijw, u_ind = np.unique(data, return_inverse=True)
	unique_ijw = unique_ijw.view(data.dtype).reshape(-1, data.shape[0]).T

	print 'Unique_ijw:', unique_ijw
	print 'u_ind:', u_ind
	return unique_ijw


# Returns the degree of a set of vertices from edge list
def degree( i ):
	unique, counts = np.unique(i, return_counts=True)
	return counts


# Returns the output degree of the left set and input degree of the right set.
def bi_degree( ij ):
	unique_i, counts_i = np.unique(ij[0,:], return_counts=True)
	unique_j, counts_j = np.unique(ij[1,:], return_counts=True)
	return counts_i, counts_j


# Utility function to print the matchings found from the matching algorithm.
# matching : dictionary with matching[v] containing vertex matched to v
def print_matching( matching ):
  i = 0
  for key, value in matching.iteritems():
    i+=1
    print str(key) + '  :  ' + str(value) + '\t',
    if i % 5 == 0:
      print ''
  print ''

'''
def print_stats( self ):
  print 'Average degree of entire graph', self.ave_degree
  print 'Average degree of left: ', self.left_degree
  print 'Average degree of right:', self.right_degree
  print 'Number of vertices in left:', self.left
  print 'Number of vertices in right:', self.right
  print 'Number of edges:', self.edges, '\n'

'''

def write_timings( left, right, edges, total, filename ):
  with open(filename, 'a') as file:
      file.write('V1: ' + str(left) + '\n')
      file.write('V2: ' + str(right) + '\n')
      file.write('Edges: ' + str(edges) + '\n')
      file.write('Total time: ' + str(total) + '\n\n')

# Reads timings from file and returns list of tuples of (verts, edges, time)
def collect_timings():
	timings = []
	with open('HKTimesCRC.txt') as infile:
		for line in infile:
			line = line[:-1]
			if line.startswith('V1'):
				v1 = line.split(' ')[1]
			elif line.startswith('V2'):
				v2 = line.split(' ')[1]
			elif line.startswith('Edges'):
				edges = line.split(' ')[1]
			elif line.startswith('Total'):
				time = float(line.split(' ')[2])
				verts = float(v1) + float(v2)
				timings.append( (verts, float(edges), time) )
	return timings


# Plots a histogram of the outdegree against the frequency
def plot_hist( counts ):
	hist = plt.figure(2)
	plt.hist( counts, bins=5000, align='left' )
	plt.suptitle('Outdegree Histogram (Non-Log)')
	plt.xlabel('Outdegree')
	plt.ylabel('Count')
	plt.show(True)


# Plots the degree distribution
def plot_points( counts, title ):
	unique_2, counts_2 = np.unique(counts, return_counts=True)
	fig = plt.figure(3)
	plt.scatter( unique_2, counts_2, s=0.4 )
	plt.suptitle( title )
	plt.xlabel('Degree')
	plt.ylabel('Frequency')
	plt.xscale('log')
	plt.yscale('log')
	plt.show(True)

# Plots the scaling of HK algorithm after reading in timings
def plot_scaling( timings ):
	veTimes = []
	for i in timings:
		ve = math.sqrt(i[0]) * i[1]
		veTimes.append((ve, i[2]))

	xval = [x[0] for x in veTimes]
	yval = [x[1] for x in veTimes]
	plt.xlabel('Vertex Count')
	plt.ylabel('Time (s)')
	plt.suptitle('Hopcroft-Karp Time Complexity')
	plt.scatter(xval, yval)
	plt.show()