#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <sched.h>
#include <unistd.h>
#include <mppaipc.h>

struct Arguments
{
  int dummy;
};

int main(int argc, char **argv){
      
  if(argc != 9){
    printf ("Wrong number of parameters.\n");
    printf("Usage: WIDTH HEIGHT TILING_HEIGHT TILING_WIDTH ITERATIONS INNER_ITERATIONS NUMBER_CLUSTERS NUMBER_THREADS\n");
    exit(-1);
  }

  int width, 
      height,
      tilingHeight,
      tilingWidth,
      iterations,
      innerIterations,
      nb_clusters,
      nb_threads;

  //Stencil configuration
  width = atoi(argv[1]);
  height = atoi(argv[2]);
  tilingHeight = atoi(argv[3]);
  tilingWidth = atoi(argv[4]);
  iterations = atoi(argv[5]);
  innerIterations = atoi(argv[6]);
  nb_clusters = atoi(argv[7]);
  nb_threads = atoi(argv[8]);

  Mask2D<float> mask(25);
  mask.set(0,-2,2,0.0); mask.set(1,-1,2,0.0); mask.set(2,0,2,0.0);  mask.set(3,1,2,0.0);  mask.set(4,2,2,0.0);
  mask.set(5,-2,1,0.0); mask.set(6,-1,1,0.0); mask.set(7,0,1,0.1);  mask.set(8,1,1,0.0);  mask.set(9,2,1,0.0);
  mask.set(10,-2,0,0.0);  mask.set(11,-1,0,0.1);  mask.set(12,0,0,0.2); mask.set(13,1,0,0.1); mask.set(14,2,0,0.0);
  mask.set(15,-2,-1,0.0); mask.set(16,-1,-1,0.0); mask.set(17,0,-1,0.1);  mask.set(18,1,-1,0.0);  mask.set(19,2,-1,0.0);
  mask.set(20,-2,-2,0.0); mask.set(21,-1,-2,0.0); mask.set(22,0,-2,0.0);  mask.set(23,1,-2,0.0);  mask.set(24,2,-2,0.0);
  
  int halo_value = mask.getRange()*innerIterations;

  Array2D<float> inputGrid(width, height, halo_value);
  Array2D<float> outputGrid(width, height, halo_value);

  srand(1234);
  for(int h = 0 + halo_value; h < height + halo_value; h++) {
    for(int w = 0 + halo_value; w < width + halo_value; w++) {
      inputGrid(h,w) = ((float)(rand() % 255))/255;
    }
  }

  // std::string grid;
  // for(int h = 0+ halo_value; h < height + halo_value; h++) {
  //  for(int w = 0+ halo_value ; w < width + halo_value;  w++) {
  //   float element = inputGrid(h,w);
  //     char celement[10];
  //     sprintf(celement, " %f", element);
  //   grid+= celement;
  //  }
  //  grid += "\n";
  // }
  // std::cout << grid << std::endl;

  // std::string grid;
  // for(int h = 0; h < height + halo_value*2; h++) {
  //  for(int w = 0; w < width + halo_value*2;  w++) {
  //    int element = inputGrid(h,w);
  //    char celement[10];
  //    sprintf(celement, " %d", element);
  //    grid+= celement;
  //  }
  //  grid += "\n";
  // }
  // std::cout << grid << std::endl;

  //Instantiate Stencil 2D
  Stencil2D<Array2D<float>, Mask2D<float>, Arguments> stencil(inputGrid, outputGrid, mask);
  
//  std::chrono::time_point<std::chrono::steady_clock> start = mppa_master_get_time();

  //Schedule computation to slaves
  stencil.scheduleMPPA("convolution-async-slave", nb_clusters, nb_threads, width, height, tilingHeight, tilingWidth, iterations, innerIterations);

//  std::chrono::time_point<std::chrono::steady_clock> end = mppa_master_get_time();
//  cout<<"Master Time: " << mppa_master_diff_time(start,end) << endl;

  // std::string grid;
  // int outter_iterations = ceil(float(iterations)/innerIterations);
  // if(outter_iterations %2 == 1) {
  //   grid = "";
  //   for(int h=0+ halo_value; h < height + halo_value; h++) {
  //    for(int w=0+ halo_value; w < width + halo_value; w++) {
  //     float element = outputGrid(h,w);
  //       char celement[10];
  //       sprintf(celement, " %f", element);
  //     grid+= celement;
  //    }
  //    grid += "\n";
  //   }
  //   std::cout << "printing output" << std::endl;
  //   std::cout << grid << std::endl;
  // } else { 
  //   grid = "";
  //   for(int h=0+ halo_value; h < height + halo_value; h++) {
  //    for(int w=0+ halo_value; w < width + halo_value; w++) {
  //     float element = inputGrid(h,w);
  //       char celement[10];
  //       sprintf(celement, " %f", element);
  //     grid+= celement;
  //    }
  //    grid += "\n";
  //   }
  //   std::cout << "printing input" << std::endl;
  //   std::cout << grid << std::endl;
  // }
    
  // grid = "";
  // for(int h=0; h < height + halo_value*2; h++) {
  //  for(int w=0; w < width + halo_value*2; w++) {
  //    int element = outputGrid(h,w);
  //    char celement[10];
  //    sprintf(celement, " %d", element);
  //    grid+= celement;
  //  }
  //  grid += "\n";
  // }
  // std::cout << grid << std::endl;
  
  // grid = "";
  // for(int h = 0; h < height + halo_value*2; h++) {
  //  for(int w = 0; w < width + halo_value*2;  w++) {
  //    int element = inputGrid(h,w);
  //    char celement[10];
  //    sprintf(celement, " %d", element);
  //    grid+= celement;
  //  }
  //  grid += "\n";
  // }
  // std::cout << grid << std::endl;


  exit(0);
}
