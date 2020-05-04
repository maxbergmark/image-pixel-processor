

all:
	g++ jpeg_test.cpp --std=c++17 -g -L/usr/lib/x86_64-linux-gnu -Wl,--no-undefined -lturbojpeg -fopenmp -O3 -lstdc++fs