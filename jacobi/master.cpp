#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <sched.h>
#include <unistd.h>
#include <mppaipc.h>

#define PSKEL_MPPA
#define MPPA_MASTER
// #define DEBUG
//#define BUG_TEST
// #define PRINT_OUT
// #define TIME_EXEC
// #define TIME_SEND
#include "../../include/PSkel.h"

using namespace std;
using namespace PSkel;

int main(int argc, char **argv){
	int width, height, tilingHeight, tilingWidth, iterations, innerIterations, nb_clusters, nb_threads; //stencil size
	if(argc != 9){
		printf ("Wrong number of parameters.\n");
		printf("Usage: WIDTH HEIGHT TILING_HEIGHT TILING_WIDTH ITERATIONS INNER_ITERATIONS NUMBER_CLUSTERS NUMBER_THREADS\n");
		exit(-1);
	}

	//Stencil configuration
	width = atoi(argv[1]);
	height = atoi(argv[2]);
  tilingHeight = atoi(argv[3]);
  tilingWidth = atoi(argv[4]);
	iterations = atoi(argv[5]);
	innerIterations = atoi(argv[6]);
	nb_clusters = atoi(argv[7]);
	nb_threads = atoi(argv[8]);

	//Mask configuration
	float factor = 1.f/(float)width;

	Mask2D<float> mask(4);

	mask.set(0,1,0,0);
	mask.set(1,-1,0,0);
	mask.set(2,0,1,0);
	mask.set(3,0,-1,0);

	int halo_value = mask.getRange()*innerIterations;

	Array2D<float> inputGrid(width,height, halo_value);
	Array2D<float> outputGrid(width,height, halo_value);

	for(size_t h=0;h<inputGrid.getRealHeight();h++) {
		for(size_t w=0;w<inputGrid.getRealWidth();w++) {
			inputGrid(h+halo_value,w+halo_value) = 1.0 + w*0.1 + h*0.01;
			// inputGrid(h,w) = h*inputGrid.getWidth() + w;
		    //printf("inputGrid(%d,%d) = %f;\n", h, w, inputGrid(h,w));
		}
	}

	// std::string grid;
 // for(int h = 0+ halo_value; h < height + halo_value; h++) {
 //   for(int w = 0 + halo_value; w < width + halo_value;  w++) {
 //   	float element = inputGrid(h,w);
	// 		char celement[30];
	// 		sprintf(celement, " %f", element);
 //   	grid+= celement;
 //   }
 //   grid += "\n";
	// }
	// std::cout << grid << std::endl;

	//Instantiate Stencil 2D
	Stencil2D<Array2D<float>, Mask2D<float>, float> stencil(inputGrid, outputGrid, mask, factor);


	//Schedule computation to slaves
	stencil.scheduleMPPA("jacobi-async-slave", nb_clusters, nb_threads, width, height, tilingHeight, tilingWidth, iterations, innerIterations);

	// grid = "";
	// for(int h=0+ halo_value; h < height + halo_value; h++) {
	//  for(int w=0+ halo_value; w < width + halo_value; w++) {
	//  	int element = outputGrid(h,w);
	// 		char celement[10];
	// 		sprintf(celement, " %d", element);
	//  	grid+= celement;
	//  }
	//  grid += "\n";
	// }
	// std::cout << grid << std::endl;

	// grid = "";
	// for(int h=0+ halo_value; h < height + halo_value; h++) {
	//  for(int w=0+ halo_value; w < width + halo_value; w++) {
	//  	int element = inputGrid(h,w);
	// 		char celement[10];
	// 		sprintf(celement, " %d", element);
	//  	grid+= celement;
	//  }
	//  grid += "\n";
	// }
	// std::cout << grid << std::endl;

	// int outter_iterations = ceil(float(iterations)/innerIterations);
	// if(outter_iterations %2 == 1) {
	// 	grid = "";
	// 	for(int h=0+ halo_value; h < height + halo_value; h++) {
	// 	 for(int w=0+ halo_value; w < width + halo_value; w++) {
	// 	 	float element = outputGrid(h,w);
	// 			char celement[30];
	// 			sprintf(celement, " %f", element);
	// 	 	grid+= celement;
	// 	 }
	// 	 grid += "\n";
	// 	}
	// 	std::cout << "printing output" << std::endl;
	// 	std::cout << grid << std::endl;
	// } else { 
	// 	grid = "";
	// 	for(int h=0+ halo_value; h < height + halo_value; h++) {
	// 	 for(int w=0+ halo_value; w < width + halo_value; w++) {
	// 	 	float element = inputGrid(h,w);
	// 			char celement[10];
	// 			sprintf(celement, " %f", element);
	// 	 	grid+= celement;
	// 	 }
	// 	 grid += "\n";
	// 	}
	// 	std::cout << "printing input" << std::endl;
	// 	std::cout << grid << std::endl;
	// }


	// grid = "";
	// for(int h=0+ halo_value; h < height + halo_value; h++) {
	//  for(int w=0+ halo_value; w < width + halo_value; w++) {
	//  	int element = outputGrid(h,w);
	// 		char celement[10];
	// 		sprintf(celement, " %d", element);
	//  	grid+= celement;
	//  }
	//  grid += "\n";
	// }
 // 	std::cout << grid << std::endl;

	exit(0);
}
