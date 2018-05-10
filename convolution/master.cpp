#include <stdio.h>   /* printf */
#include <stdlib.h>  /* exit   */
#include <math.h>    /* ceil   */
#include <assert.h>  /* assert */ 
#include <utask.h>
#include <mppa_async.h>
#include <mppa_power.h>

#include <iostream>

#define ARGC_SLAVE 14

int main(int argc, char **argv)
{
if(argc != 9){
    printf ("Wrong number of parameters.\n");
    printf("Usage: WIDTH HEIGHT TILING_HEIGHT TILING_WIDTH ITERATIONS INNER_ITERATIONS NUMBER_CLUSTERS NUMBER_THREADS\n");
    exit(-1);
}

/* Args reading */
int width            = atoi(argv[1]);
int height           = atoi(argv[2]);
int tiling_height    = atoi(argv[3]);
int tiling_width     = atoi(argv[4]);
int iterations       = atoi(argv[5]);
int inner_iterations = atoi(argv[6]);
int nb_clusters      = atoi(argv[7]);
int nb_threads       = atoi(argv[8]);

int  mask_range;
int  halo_value;
int  width_enlarged;
int  height_enlarged;    
float *input_grid;
float *output_grid;
mppa_async_segment_t input_mppa_segment;
mppa_async_segment_t output_mppa_segment;

/* Initializing arrays */
mask_range = 2;
halo_value = mask_range * inner_iterations;
width_enlarged = width + (halo_value * 2);
height_enlarged = height + (halo_value * 2);

input_grid = (float *)calloc(width_enlarged * height_enlarged, sizeof(float));
output_grid = (float *)calloc(width_enlarged * height_enlarged, sizeof(float));

srand(1234);
for(int h = 0 + halo_value; h < height + halo_value; h++)
    for(int w = 0 + halo_value; w < width + halo_value; w++)
        input_grid[h * width_enlarged + w] = ((float)(rand() % 255))/255;

/* Prepare to spawn clusters */
size_t w_tiling = ceil(float(width)/float(tiling_width));
size_t h_tiling = ceil(float(height)/float(tiling_height));
size_t total_size = float(h_tiling*w_tiling);

/* MPPA initialization */
mppa_rpc_server_init(1, 0, (int)total_size < nb_clusters ? total_size : nb_clusters);
mppa_async_server_init();

int tiles = total_size/nb_clusters;
int it_mod = total_size % nb_clusters;
int outter_iterations = ceil(float(iterations)/inner_iterations);

char **argv_slave = (char**) malloc(sizeof (char*) * ARGC_SLAVE);
for(auto i = 0; i < ARGC_SLAVE; ++i){
    argv_slave[i] = (char*) malloc (sizeof (char) * 10);
}

/* Prepare clusters arguments */
sprintf(argv_slave[2],  "%d", tiling_width);
sprintf(argv_slave[3],  "%d", tiling_height);
sprintf(argv_slave[5],  "%d", nb_threads);
sprintf(argv_slave[6],  "%d", inner_iterations);
sprintf(argv_slave[7],  "%d", outter_iterations);
sprintf(argv_slave[8],  "%d", it_mod);
sprintf(argv_slave[9],  "%d", nb_clusters);
sprintf(argv_slave[10], "%d", width);
sprintf(argv_slave[11], "%d", height);
argv_slave[13] = NULL;

int r;
int cluster_id;
int tiles_slave;
int nb_computated_tiles = 0;

/* Create a task to manage the rpc server on another processor */
utask_t t;
utask_create(&t, NULL, (void* (*)(void*))mppa_rpc_server_start, NULL);

assert(mppa_async_segment_create(&input_mppa_segment,  1, input_grid,  width_enlarged * height_enlarged * sizeof(float), 0, 0, NULL) == 0);
assert(mppa_async_segment_create(&output_mppa_segment, 2, output_grid, width_enlarged * height_enlarged * sizeof(float), 0, 0, NULL) == 0);

/* Loop to cluster initialization */
for (cluster_id = 0; cluster_id < nb_clusters && cluster_id < (int)total_size; cluster_id++) {
    r = (cluster_id < it_mod)?1:0;
    tiles_slave = tiles + r;

    sprintf(argv_slave[1],  "%d", tiles_slave);
    sprintf(argv_slave[4],  "%d", cluster_id);
    sprintf(argv_slave[12], "%d", nb_computated_tiles);
    
    nb_computated_tiles += tiles_slave;

    if (mppa_power_base_spawn(cluster_id, "convolution-async-slave", (const char **)argv_slave, NULL, MPPA_POWER_SHUFFLING_ENABLED) == -1)
        printf("# [IODDR0] Fail to Spawn cluster %d\n", cluster_id);
}


/* Wait the end of the clusters */
int status = 0;
for(cluster_id = 0; cluster_id < nb_clusters && cluster_id < (int)total_size; cluster_id++){
    int ret;
    if (mppa_power_base_waitpid(cluster_id, &ret, 0) < 0) {
        printf("# [IODDR0] Waitpid failed on cluster %d\n", cluster_id);
    }
    status += ret;
}

if(status != 0)
    exit(-1);

/* House keeping */
assert(mppa_async_segment_destroy(&input_mppa_segment)  == 0);
assert(mppa_async_segment_destroy(&output_mppa_segment) == 0);
for (auto i = 0; i < ARGC_SLAVE; ++i)
    free(argv_slave[i]);
free(argv_slave);
free(input_grid);
free(output_grid);

return 0;
}
