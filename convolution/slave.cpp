#include <stdio.h>   /* printf    */
#include <unistd.h>  /* exit      */
#include <math.h>    /* ceil      */
#include <assert.h>  /* assert    */ 
#include <iostream>  /* std::cout */
#include <omp.h>
#include <mppa_async.h>

#include "mppa_utils.h"

static float h_constant; 

void stencil_kernel(float *input, float *output, int width, int i, int j)
{
    output[i*width + j] = 
                          // input[(i-2)*width + (j+2)] * 0.0 +
                          // input[(i-1)*width + (j+2)] * 0.0 +
                          // input[ i   *width + (j+2)] * 0.0 +
                          // input[(i+1)*width + (j+2)] * 0.0 +
                          // input[(i+2)*width + (j+2)] * 0.0 +
                          // input[(i-2)*width + (j+1)] * 0.0 +
                          // input[(i-1)*width + (j+1)] * 0.0 +
                          input[ i   *width + (j+1)] * 0.1 +
                          // input[(i+1)*width + (j+1)] * 0.0 +
                          // input[(i+2)*width + (j+1)] * 0.0 +
                          // input[(i-2)*width +  j   ] * 0.0 +
                          input[(i-1)*width +  j   ] * 0.1 +
                          input[ i   *width +  j   ] * 0.2 +
                          input[(i+1)*width +  j   ] * 0.1 +
                          // input[(i+2)*width +  j   ] * 0.0 +
                          // input[(i-2)*width + (j-1)] * 0.0 +
                          // input[(i-1)*width + (j-1)] * 0.0 +
                          input[ i   *width + (j-1)] * 0.1;
                          // input[(i+1)*width + (j-1)] * 0.0 +
                          // input[(i+2)*width + (j-1)] * 0.0 +
                          // input[(i-2)*width + (j-2)] * 0.0 +
                          // input[(i-1)*width + (j-2)] * 0.0 +
                          // input[ i   *width + (j-2)] * 0.0 +
                          // input[(i+1)*width + (j-2)] * 0.0 +
                          // input[(i+2)*width + (j-2)] * 0.0; 
}

void run_openmp(float *in, float *out, int width, int nb_threads, struct work_area_t *work_area)
{
    omp_set_num_threads(nb_threads);
    #pragma omp parallel for
    for (auto h = work_area->y_init; h < work_area->y_final; h++)
        for (auto w = work_area->x_init; w < work_area->x_final; w++)
            stencil_kernel(in, out, width, h, w);
}

int main(int argc,char **argv)
{
mppa_rpc_client_init();
mppa_async_init();

int nb_tiles            = atoi(argv[1]);
int tilling_width       = atoi(argv[2]);
int tilling_height      = atoi(argv[3]);
int nb_threads          = atoi(argv[5]);
int inner_iterations    = atoi(argv[6]);
int outter_iterations   = atoi(argv[7]);
int width               = atoi(argv[10]);
int height              = atoi(argv[11]);
int nb_computated_tiles = atoi(argv[12]);

float *input_grid;
float *output_grid;
mppa_async_segment_t* input_mppa_segment;
mppa_async_segment_t* output_mppa_segment;

/* Initializing arrays */
int mask_range = 2;
int halo_value = mask_range * inner_iterations;
int width_enlarged = tilling_width + (halo_value * 2);
int height_enlarged = tilling_height + (halo_value * 2);

input_grid  = (float*)calloc(width_enlarged * height_enlarged, sizeof(float));
output_grid = (float*)calloc(width_enlarged * height_enlarged, sizeof(float));
input_mppa_segment  = new mppa_async_segment_t();
output_mppa_segment = new mppa_async_segment_t();

h_constant = 4.f / (float) (width*width);

double exec_time = 0;
double comm_time = 0;
double clear_time = 0;
double segment_swap_time = 0;
double work_area_time = 0;
double computation_time = 0;
double barrier_time = 0;

mppa_rpc_barrier_all();

auto begin_exec = mppa_slave_get_time();
auto begin_comm = mppa_slave_get_time();

assert(mppa_async_segment_clone(input_mppa_segment,  1, NULL, 0, NULL) == 0);
assert(mppa_async_segment_clone(output_mppa_segment, 2, NULL, 0, NULL) == 0);

/* Number of tiles in x axis */
int w_tiling = ceil(float(width)/float(tilling_width));

/* Starting point in x axis */
int width_offset = (nb_computated_tiles % w_tiling) * (tilling_width);

/* Startig point in y axis */
int height_offset = floor(float(nb_computated_tiles)/float(w_tiling)) * (tilling_height);

auto end_comm = mppa_slave_get_time();
comm_time += mppa_slave_diff_time(begin_comm, end_comm);

struct work_area_t* work_area = new work_area_t(0, 0, 0, 0, {0,0,0,0});


for(int out_iteration = 0; out_iteration < outter_iterations; ++out_iteration){
    int nb_tiles_aux = nb_tiles;
    int j = width_offset;
    for(int i = height_offset; i < height && nb_tiles_aux; i+=tilling_height){
        for(; j < width && nb_tiles_aux; j+=tilling_width){
            auto work_area_time_aux = mppa_slave_get_time();
            
            work_area->x_init  = mask_range; 
            work_area->y_init  = mask_range; 
            work_area->x_final = width_enlarged  - mask_range; 
            work_area->y_final = height_enlarged - mask_range; 
            work_area->dist_to_border = {0,0,0,0};

            /* Top border */
            if (i - halo_value < 0) {
                work_area->y_init = -(i-(halo_value));
                work_area->dist_to_border[0] = -(i - (halo_value)) - mask_range;
            }
            /* Rigth border*/
            if (j + tilling_width + (halo_value) - width  > 0) {
                work_area->x_final = width_enlarged - (j + tilling_width + (halo_value) - width);
                work_area->dist_to_border[1] = (j + tilling_width + (halo_value) - width) - mask_range;
            }
            /* Bottom border */
            if (i + tilling_height + (halo_value) - height > 0) {
                work_area->y_final = height_enlarged - (i + tilling_height + (halo_value) - height);
                work_area->dist_to_border[2] = (i + tilling_height + (halo_value) - height) - mask_range;
            }
            /* Left border */
            if(j - (halo_value) < 0) {
                work_area->x_init = -(j-(halo_value));
                work_area->dist_to_border[3] = (-(j-(halo_value))) - mask_range;
            }
  
            work_area_time += mppa_slave_diff_time(work_area_time_aux, mppa_slave_get_time());
  
            auto clear_aux = mppa_slave_get_time();
            memset(output_grid, 0, width_enlarged * height_enlarged * sizeof(float));
            clear_time += mppa_slave_diff_time(clear_aux, mppa_slave_get_time());
  
            begin_comm = mppa_slave_get_time();

  
            mppa_async_point2d_t remote_point = {
                j, // xpos
                i, // ypos
                width + halo_value*2, // xdim
                height + halo_value*2, // ydim
            };
  
            mppa_async_point2d_t local_point = 
            {
              0,               // xpos
              0,               // ypos
              width_enlarged,  // xdim
              height_enlarged, // ydim
            };
  
            assert(mppa_async_sget_block2d(input_grid, 
                                           input_mppa_segment,
                                           0, sizeof(float), width_enlarged, height_enlarged,
                                           &local_point,
                                           &remote_point,
                                           NULL) == 0);
  

            end_comm = mppa_slave_get_time();
            comm_time += mppa_slave_diff_time(begin_comm, end_comm);
  
            auto computation_time_aux = mppa_slave_get_time();
            
            for(int i = 0; i < inner_iterations; i++) {
                if(i%2==0) {
                    run_openmp(input_grid, output_grid, width_enlarged, nb_threads, work_area);
                } else {
                    run_openmp(output_grid, input_grid, width_enlarged, nb_threads, work_area);
                }
  
                /* Control top border */
                if(!work_area->dist_to_border[0]) {
                    work_area->y_init += mask_range;
                } else {
                    work_area->dist_to_border[0] -= mask_range;
                }
                /* Control right border */
                if(!work_area->dist_to_border[1]) {
                    work_area->x_final -= mask_range;
                } else {
                    work_area->dist_to_border[1] -= mask_range;
                }
                /* Control bottom border */
                if(!work_area->dist_to_border[2]) {
                    work_area->y_final -= mask_range;
                } else {
                    work_area->dist_to_border[2] -= mask_range;
                }
                /* Control left border */
                if(!work_area->dist_to_border[3]) {
                    work_area->x_init += mask_range;
                } else {
                    work_area->dist_to_border[3] -= mask_range;
                }
            }
  
            computation_time += mppa_slave_diff_time(computation_time_aux, mppa_slave_get_time());
  
            begin_comm = mppa_slave_get_time();

            remote_point.xpos += halo_value;
            remote_point.ypos += halo_value;
            
            if(inner_iterations % 2 == 0) {
                auto swap_time_aux = mppa_slave_get_time();
                mppa_async_segment_t *aux = input_mppa_segment;
                input_mppa_segment = output_mppa_segment;
                
                mppa_async_point2d_t local_point = {
                    0 + halo_value,  // xpos
                    0 + halo_value,  // ypos
                    width_enlarged,  // xdim
                    height_enlarged, // ydim
                };

                assert(mppa_async_sput_block2d(input_grid, 
                                               input_mppa_segment,
                                               0, sizeof(float), tilling_width, tilling_height,
                                               &local_point,
                                               &remote_point,
                                               NULL) == 0);
  
                input_mppa_segment = aux;
                segment_swap_time += mppa_slave_diff_time(swap_time_aux, mppa_slave_get_time());
            } else {
                mppa_async_point2d_t local_point = {
                    0 + halo_value,  // xpos
                    0 + halo_value,  // ypos
                    width_enlarged,  // xdim
                    height_enlarged, // ydim
                };

                assert(mppa_async_sput_block2d(output_grid, 
                                               output_mppa_segment,
                                               0, sizeof(float), tilling_width, tilling_height,
                                               &local_point,
                                               &remote_point,
                                               NULL) == 0);
            } 

            end_comm = mppa_slave_get_time();
            comm_time += mppa_slave_diff_time(begin_comm, end_comm);
  
            --nb_tiles_aux;
        }
        j = 0; /* "reset" ypos of tile */
    }

    auto barrier_time_aux = mppa_slave_get_time();
    mppa_rpc_barrier_all();
    barrier_time += mppa_slave_diff_time(barrier_time_aux, mppa_slave_get_time());

    auto swap_time_aux_2 = mppa_slave_get_time();
    
    /* Segments swap */
    mppa_async_segment_t* aux = input_mppa_segment;
    input_mppa_segment = output_mppa_segment;
    output_mppa_segment = aux;

    segment_swap_time += mppa_slave_diff_time(swap_time_aux_2, mppa_slave_get_time());

}

auto end_exec = mppa_slave_get_time();
exec_time = mppa_slave_diff_time(begin_exec, end_exec);

std::cout<< "Slave Time: " << exec_time << std::endl;
std::cout<< "Comm. Time: " << comm_time << std::endl;
std::cout<< "Clear Time: " << clear_time << std::endl;
std::cout<< "Swap Time: " << segment_swap_time << std::endl;
std::cout<< "Work_Area Time: " << work_area_time << std::endl;
std::cout<< "Computation Time: " << computation_time << std::endl;
std::cout<< "Barrier Time: " << barrier_time << std::endl;

free(input_grid);
free(output_grid);
delete work_area;
delete input_mppa_segment;
delete output_mppa_segment;

mppa_async_final();
return 0;
}
