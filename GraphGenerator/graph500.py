import numpy as np
import random
import scipy.sparse as sp
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import math

class Graph:
  def __init__(self, adjList, left, right, edges):
    self.adjList = adjList
    self.left = left
    self.right = right
    self.edges = edges


def kronecker_generator( scale, edgeFactor ):
	N = 2 ** scale
	M = edgeFactor * N

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


def bipartite_kronecker( scale_1, scale_2, edgeFactor ):
	N = 2 ** scale_1
	M = (edgeFactor * N) / 2

	A_I = 0.57 #0.57
	B_I = 0.19 #0.10
	C_I = 0.19 #0.19

	ab_I = A_I + B_I
	c_norm_I = C_I/(1 - (A_I + B_I))
	a_norm_I = A_I/(A_I + B_I)


	P = 2 ** scale_2

	A_O = 0.57
	B_O = 0.19
	C_O = 0.19

	ab_O = A_O + B_O
	c_norm_O = C_O/(1 - (A_O + B_O))
	a_norm_O = A_O/(A_O + B_O)

	i = np.zeros(M)
	j = np.zeros(M)

	for c in range(scale_1):
		i_bit = np.random.rand(1, M) > ab_I
		j_bit = np.greater( np.random.rand(1, M), ((c_norm_I * i_bit) + (a_norm_I * ~i_bit)) )
		i = i - 2**(c) * i_bit
		j = j - 2**(c) * j_bit

	i = i - 1
	j = j - 1

	ij = np.append(i, j, axis=1)
	print 'MIN OF IJ:', np.min(ij)
	print 'MAX OF IJ:', np.max(ij)

	a = np.ones(M)
	b = np.ones(M)

	for d in range(scale_2):
		a_bit = np.random.rand(1, M) > ab_O
		b_bit = np.greater( np.random.rand(1, M), ((c_norm_O * a_bit) + (a_norm_O * ~a_bit)) )
		a = a + 2**(d) * a_bit
		b = b + 2**(d) * b_bit

	a = a - 1
	b = b - 1

	ab = np.append(a, b, axis=1)
	print 'MIN OF AB:', np.min(ab)
	print 'MAX OF AB:', np.max(ab)

	edgeList = np.ones((3,2*M))
	edgeList[0, :] = ij
	edgeList[1, :] = ab
	edgeList[2, :] = np.random.rand(1, 2*M)

	perm_N = np.random.permutation(N)
	edgeList[0, :] = 0 - perm_N[ edgeList[0, :].astype(int) ]
	print edgeList[0]

	perm_P = np.random.permutation(P)
	edgeList[1, :] = perm_P[ edgeList[1, :].astype(int) ]

	perm = np.random.permutation(2*M)
	edgeList = edgeList[:, perm]

	return edgeList	


def bipartite_noisy_kronecker( scale_1, scale_2, edgeFactor ):
	N = 2 ** scale_1
	P = 2 ** scale_2
	M = (edgeFactor * N) / 2
	noise = 0.05

	A_I = 0.46 #0.57
	B_I = 0.19 #0.10
	C_I = 0.19 #0.19
	D_I = 0.05

	A_O = 0.46
	B_O = 0.19
	C_O = 0.19
	D_O = 0.05

	i = np.ones(M)
	j = np.ones(M)

	for c in range(scale_1):
		ab_I = A_I + B_I
		c_norm_I = C_I/(1 - (A_I + B_I))
		a_norm_I = A_I/(A_I + B_I)
		u = random.uniform( -noise, noise )
		A_I = A_I - ((2 * u * A_I)/(A_I + D_I))
		B_I = B_I + u
		C_I = C_I + u
		D_I = D_I - ((2 * u * D_I)/(A_I + D_I))

		i_bit = np.random.rand(1, M) > ab_I
		j_bit = np.greater( np.random.rand(1, M), ((c_norm_I * i_bit) + (a_norm_I * ~i_bit)) )
		i = i + 2**(c) * i_bit
		j = j + 2**(c) * j_bit

	i = i - 1
	j = j - 1

	ij = np.append(i, j, axis=1)
	print 'TOTAL RANGE FOR GRAPH: ', np.ptp(ij)
	a = np.ones(M)
	b = np.ones(M)

	for d in range(scale_2):
		ab_O = A_O + B_O
		c_norm_O = C_O/(1 - (A_O + B_O))
		a_norm_O = A_O/(A_O + B_O)
		u = random.uniform( -noise, noise )
		A_O = A_O - ((2 * u * A_O)/(A_O + D_O))
		B_O = B_O + u
		C_O = C_O + u
		D_O = D_O - ((2 * u * D_O)/(A_O + D_O))

		a_bit = np.random.rand(1, M) > ab_O
		b_bit = np.greater( np.random.rand(1, M), ((c_norm_O * a_bit) + (a_norm_O * ~a_bit)) )
		a = a + 2**(d) * a_bit
		b = b + 2**(d) * b_bit

	a = a - 1
	b = b - 1

	ab = np.append(a, b, axis=1)

	edgeList = np.ones((3,2*M))
	edgeList[0, :] = ij
	edgeList[1, :] = ab
	edgeList[2, :] = np.random.rand(1, 2*M)

	perm_N = np.random.permutation(N)
	edgeList[0, :] = -1 - perm_N[ edgeList[0, :].astype(int) ]

	perm_P = np.random.permutation(P)
	edgeList[1, :] = perm_P[ edgeList[1, :].astype(int) ]

	perm = np.random.permutation(2*M)
	edgeList = edgeList[:, perm]

	return edgeList



def bipartiteMatch( graph ):
	'''
	Find maximum cardinality matching of a bipartite graph (U,V,E).
	The input format is a dictionary mapping members of U to a list
	of their neighbors in V.  The output is a triple (M,A,B) where M is a
	dictionary mapping members of V to their matches in U, A is the part
	of the maximum independent set in U, and B is the part of the MIS in V.
	The same object may occur in both U and V, and is treated as two
	distinct vertices if this happens.
	'''

	# initialize greedy matching (redundant, but faster than full search)
	######################## O(E) ########################
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


def toAdjList( edgeList ):
	adjList = defaultdict(set)
	for i in edgeList.T:
		adjList[int(i[0])].add(int(i[1]))

	for i in adjList:
		adjList[i] = list(adjList[i])

	return adjList


def writeAdjList( graph, left, right, edges, filename ):
  with open(filename, 'w') as outfile:
    outfile.write(str(left) + ',' + str(right) + ',' + str(edges) + '\n')
    for i in graph:
      outfile.write(str(i) + ' ' + str(graph[i]) + '\n')


def writeAdjMat( graph, filename ):
	with open(filename, 'w') as outfile:
		print 'Need biadjacency matrix, unless the other side can convert'
		#figure it out


def count_edges( adjList ):
	count = sum(len(v) for v in adjList.itervalues())
	return count


def make_undirected( ijw ):
	jiw = np.asarray([ijw[1,:], ijw[0,:], ijw[2,:]])
	ijw = np.append(ijw, jiw, axis=1)
	return ijw


def remove_self( ijw ):
	ijw = ijw[:, ijw[0,:] != ijw[1,:]]
	return ijw


def remove_duplicate( ijw ):
	dt = np.dtype((np.void, ijw.dtype.itemsize * ijw.shape[0]))
	data = np.asfortranarray( ijw ).view(dt)
	unique_ijw, u_ind = np.unique(data, return_inverse=True)
	unique_ijw = unique_ijw.view(data.dtype).reshape(-1, data.shape[0]).T

	print 'Unique_ijw:', unique_ijw
	print 'u_ind:', u_ind
	return unique_ijw


def degree( i ):
	unique, counts = np.unique(i, return_counts=True)
	print 'Number of Vertices: ', counts.size
	return counts


def biDegree( ij ):
	unique_i, counts_i = np.unique(ij[0,:], return_counts=True)
	unique_j, counts_j = np.unique(ij[1,:], return_counts=True)
	return counts_i, counts_j


def plotSparse( G, scale ):
	N = 2**scale
	if not isinstance( G, sp.coo_matrix ):
		mat = sp.coo_matrix( (G[2], (G[0], G[1])), shape=(N,N) )
	patt = plt.figure(1)
	ax = patt.add_subplot()
	ax.plot(mat.col, mat.row, 'o', color='blue', ms=0.2)
	ax.set_xlim(0, mat.shape[1])
	ax.set_ylim(0, mat.shape[0])
	ax.set_facecolor('xkcd:white')
	ax.set_aspect('equal')
	for spine in ax.spines.values():
		spine.set_visible(False)
	ax.invert_yaxis()
	ax.set_aspect('equal')
	plt.suptitle('Sparse Matrix Pattern (Python)')
	plt.show(True)


def plotHist( counts ):
	hist = plt.figure(2)
	plt.hist( counts, bins=5000, align='left' )
	plt.suptitle('Outdegree Histogram (Non-Log)')
	plt.xlabel('Outdegree')
	plt.ylabel('Count')
	plt.show(True)


def plotPoints( counts, title ):
	unique_2, counts_2 = np.unique(counts, return_counts=True)
	fig = plt.figure(3)
	plt.scatter( unique_2, counts_2, s=0.4 )
	plt.suptitle( title )
	plt.xlabel('Degree')
	plt.ylabel('Frequency')
	plt.xscale('log')
	plt.yscale('log')
	plt.show(True)
	
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

def plotScaling( timings ):
	veTimes = []
	for i in timings:
		ve = math.sqrt(i[0]) * i[1]
		veTimes.append((ve, i[2]))
	'''
	for i in timings:
		ve = i[0]
		veTimes.append((ve, i[2]))
	'''
	xval = [x[0] for x in veTimes]
	yval = [x[1] for x in veTimes]
	plt.xlabel('Vertex Count')
	plt.ylabel('Time (s)')
	plt.suptitle('Hopcroft-Karp Time Complexity')
	plt.scatter(xval, yval)
	plt.show()
