#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define PSKEL_MPPA
#define MPPA_SLAVE
// #define DEBUG
//#define BUG_TEST
// #define PRINT_OUT
// #define TIME_EXEC
// #define TIME_SEND
// #define BARRIER_SYNC_MASTER "/mppa/sync/128:1"
// #define BARRIER_SYNC_SLAVE "/mppa/sync/[0..15]:2"
//#include "common.h"
#include "../../include/mppa_utils.h"
#include "../../include/PSkel.h"

using namespace std;
using namespace PSkel;

struct Arguments{
	float h;
};

namespace PSkel{
  __parallel__ void stencilKernel(Array2D<float> &input,Array2D<float> &output,Mask2D<float> mask, Arguments args, size_t i, size_t j){
      //printf("MaskGet(0): %f\n", mask.get(0, input, h, w));
      //printf("MaskGet(1): %f\n", mask.get(1, input, h, w));
      //printf("MaskGet(2): %f\n", mask.get(2, input, h, w));
      //printf("MaskGet(3): %f\n", mask.get(3, input, h, w));
      //printf("factor: %f\n", factor);

      // output(h,w) = 0.25*(mask.get(0,input,h,w) + mask.get(1,input,h,w) +
      //   mask.get(2,input,h,w) + mask.get(3,input,h,w) - 4*factor*factor );

      //printf("OutputValor: %f\n", output(h,w));

	    output(i,j) = 0.25f * ( input(i-1,j) + (input(i,j-1) + input(i,j+1)) + input(i+1,j) - args.h);
  }
}

int main(int argc,char **argv) {

   /**************Mask for test porpuses****************/


  Mask2D<float> mask(4);

  mask.set(0,1,0,0);
  mask.set(1,-1,0,0);
  mask.set(2,0,1,0);
  mask.set(3,0,-1,0);

  /*************************************************/

  // width = atoi (argv[1]);
  // height = atoi (argv[2]);
  // iterations = atoi (argv[3]);
  // mode = atoi(argv[4]);
  // GPUBlockSize = atoi(argv[5]);
  // numCPUThreads = atoi(argv[6]);
  // tileHeight = atoi(argv[7]);
  // tileIterations = atoi(argv[8]);

  int nb_tiles = atoi(argv[0]);
  int tilling_width = atoi(argv[1]);
  int tilling_height = atoi(argv[2]);
  int cluster_id = atoi(argv[3]);
  int nb_threads = atoi(argv[4]);
  int iterations = atoi(argv[5]);
  int outteriterations = atoi(argv[6]);
  int itMod = atoi(argv[7]);
  int nb_clusters = atoi(argv[8]);
  int width = atoi(argv[9]);
  int height = atoi(argv[10]);
  int nb_computated_tiles = atoi(argv[11]);
  
  int halo_value = mask.getRange() * iterations;

  // float factor = 1.f/(float)realWidth;

  Arguments args;
  	//args.h = 1.f / (float) x_max;
  args.h = 4.f / (float) (width*width);

  Array2D<float> partInput(tilling_width, tilling_height, halo_value);
  Array2D<float> output(tilling_width, tilling_height, halo_value);
  Stencil2D<Array2D<float>, Mask2D<float>, Arguments> stencil(partInput, output, mask, args);
  // if(iterations == 0)  {

  stencil.runMPPA(cluster_id, nb_threads, nb_tiles, outteriterations, iterations, itMod, nb_clusters, width, height, nb_computated_tiles);
  // } else {
  //      stencil.runIterativeMPPA(cluster_id, nb_threads, nb_tiles, iterations);
  //}


}
