CFLAGS = -Wall

HKC: clean Vector.o Graph.o HopKarp.o HKC.o 
	gcc $(CFLAGS) Vector.o Graph.o HopKarp.o HKC.o -o HKC

HKC.o: HKC.c HopKarp.h Graph.h Vector.h
	gcc $(CFLAGS) -c HKC.c

HopKarp.o: HopKarp.c Graph.h Vector.h
	gcc $(CFLAGS) -c HopKarp.c

Graph.o: Graph.c Vector.h
	gcc $(CFLAGS) -c Graph.c

Vector.o: Vector.c
	gcc $(CFLAGS) -c Vector.c

clean:
	rm -f HKC HKC.o HopKarp.o Vector.o Graph.o

run:
	./HKC $(args)
