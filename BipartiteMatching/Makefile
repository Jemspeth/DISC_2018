CFLAGS = -Wall -fopenmp -O1

HKC: clean Queue.o Vector.o Graph.o HopKarp.o HKC.o 
	gcc $(CFLAGS) Queue.o Vector.o Graph.o HopKarp.o HKC.o -o HKC

HKC.o: HKC.c HopKarp.h Graph.h Vector.h Queue.h
	gcc $(CFLAGS) -c HKC.c

HopKarp.o: HopKarp.c Graph.h Vector.h Queue.h
	gcc $(CFLAGS) -c HopKarp.c

Graph.o: Graph.c Vector.h Queue.h
	gcc $(CFLAGS) -c Graph.c

Vector.o: Vector.c
	gcc $(CFLAGS) -c Vector.c

Queue.o: Queue.c
	gcc $(CFLAGS) -c Queue.c

clean:
	rm -f HKC HKC.o HopKarp.o Vector.o Graph.o Queue.o

run:
	./HKC $(args)
