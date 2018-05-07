
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#define PSKEL_MPPA
#define MPPA_SLAVE

#include "../../include/mppa_utils.h"
#include "../../include/PSkel.h"

#define MASK_RADIUS 2
#define MASK_WIDTH  5

using namespace std;
using namespace PSkel;

struct Coeff{
  int null;
};

struct Arguments
{
  int dummy;
};

//*******************************************************************************************
// WB_IMAGE
//*******************************************************************************************
struct wbImage_t
{
    int  _imageWidth;
    int  _imageHeight;
    int  _imageChannels;
  PSkel::Array2D<float> _red;
  PSkel::Array2D<float> _green;
  PSkel::Array2D<float> _blue;

    wbImage_t(int imageWidth = 0, int imageHeight = 0, int imageChannels = 0) :_imageWidth(imageWidth), _imageHeight(imageHeight), _imageChannels(imageChannels)
    {
    new (&_red) PSkel::Array2D<float>(_imageWidth,_imageHeight);
    new (&_green) PSkel::Array2D<float>(_imageWidth,_imageHeight);
    new (&_blue) PSkel::Array2D<float>(_imageWidth,_imageHeight);
    }
};

int wbImage_getWidth(const wbImage_t& image){
    return image._imageWidth;
}

int wbImage_getHeight(const wbImage_t& image){
    return image._imageHeight;
}

int wbImage_getChannels(const wbImage_t& image){
    return image._imageChannels;
}

wbImage_t wbImport(char* inputFile){
    wbImage_t image;
    image._imageChannels = 3;

    std::ifstream fileInput;
    fileInput.open(inputFile, std::ios::binary);
    if (fileInput.is_open()) {
        char magic[2];
        fileInput.read(magic, 2);
        if (magic[0] != 'P' || magic[1] !='6') {
            std::cout << "expected 'P6' but got " << magic[0] << magic[1] << std::endl;
            exit(1);
        }
        char tmp = fileInput.peek();
        while (isspace(tmp)) {
            fileInput.read(&tmp, 1);
            tmp = fileInput.peek();
        }
        // filter image comments
        if (tmp == '#') {
            fileInput.read(&tmp, 1);
            tmp = fileInput.peek();
            while (tmp != '\n') {
                fileInput.read(&tmp, 1);
                tmp = fileInput.peek();
            }
        }
        // get rid of whitespaces
        while (isspace(tmp)) {
            fileInput.read(&tmp, 1);
            tmp = fileInput.peek();
        }

        //read dimensions (TODO add error checking)
        char widthStr[64], heightStr[64], numColorsStr[64], *p;
        p = widthStr;
        if(isdigit(tmp)) {
            while(isdigit(*p = fileInput.get())) {
                p++;
            }
            *p = '\0';
            image._imageWidth = atoi(widthStr);
            //std::cout << "Width: " << image._imageWidth << std::endl;
            p = heightStr;
            while(isdigit(*p = fileInput.get())) {
                p++;
            }
            *p = '\0';
            image._imageHeight = atoi(heightStr);
            //std::cout << "Height: " << image._imageHeight << std::endl;
            p = numColorsStr;
            while(isdigit(*p = fileInput.get())) {
                p++;
            }
            *p = '\0';
            int numColors = atoi(numColorsStr);
            //std::cout << "Num colors: " << numColors << std::endl;
            if (numColors != 255) {
                std::cout << "the number of colors should be 255, but got " << numColors << std::endl;
                exit(1);
            }
        } else  {
            std::cout << "error - cannot read dimensions" << std::endl;
        }

        int dataSize = image._imageWidth*image._imageHeight*image._imageChannels;
        unsigned char* data = new unsigned char[dataSize];
        fileInput.read((char*)data, dataSize);
        //float* floatData = new float[dataSize];
        
    new (&(image._red)) PSkel::Array2D<float>(image._imageWidth,image._imageHeight);
    new (&(image._green)) PSkel::Array2D<float>(image._imageWidth,image._imageHeight);
    new (&(image._blue)) PSkel::Array2D<float>(image._imageWidth,image._imageHeight);
    
    for (int y = 0; y < image._imageHeight; y++){
      for (int x = 0; x < image._imageWidth; x++){
          image._red(x,y) =   1.0*data[(y*image._imageWidth + x)*3 + 0]/255.0f;
          image._green(x,y) = 1.0*data[(y*image._imageWidth + x)*3 + 1]/255.0f;
          image._blue(x,y) =  1.0*data[(y*image._imageWidth + x)*3 + 2]/255.0f;
          
          /*if(x==1000 && y==1000){
            cout<<(int)data[(y*image._imageWidth + x)*3 + 0]<<" ";
            cout<<(int)data[(y*image._imageWidth + x)*3 + 1]<<" ";
            cout<<(int)data[(y*image._imageWidth + x)*3 + 2]<<endl;
          }*/
      }
    }
        fileInput.close();
    } else  {
         std::cout << "cannot open file " << inputFile;
         exit(1);
    }
    return image;
}

wbImage_t wbImage_new(int imageWidth, int imageHeight, int imageChannels)
{
    wbImage_t image(imageWidth, imageHeight, imageChannels);
    return image;
}

void wbImage_save(wbImage_t& image, char* outputfile) {
    std::ofstream outputFile(outputfile, std::ios::binary);
    char buffer[64];
    std::string magic = "P6\n";
    outputFile.write(magic.c_str(), magic.size());
    std::string comment  =  "# image generated by applying convolution\n";
    outputFile.write(comment.c_str(), comment.size());
    //write dimensions
    sprintf(buffer,"%d", image._imageWidth);
    outputFile.write(buffer, strlen(buffer));
    buffer[0] = ' ';
    outputFile.write(buffer, 1);
    sprintf(buffer,"%d", image._imageHeight);
    outputFile.write(buffer, strlen(buffer));
    buffer[0] = '\n';
    outputFile.write(buffer, 1);
    std::string colors = "255\n";
    outputFile.write(colors.c_str(), colors.size());

    int dataSize = image._imageWidth*image._imageHeight*image._imageChannels;
    unsigned char* rgbData = new unsigned char[dataSize];

  
  for (int y = 0; y < image._imageHeight; y++){   
    for (int x = 0; x < image._imageWidth; x++){
      rgbData[(y*image._imageWidth + x)*3 + 0] = ceil(image._red(x,y) * 255);
      rgbData[(y*image._imageWidth + x)*3 + 1] = ceil(image._green(x,y) * 255);
      rgbData[(y*image._imageWidth + x)*3 + 2] = ceil(image._blue(x,y) * 255);
    } 
  }

  outputFile.write((char*)rgbData, dataSize);
    delete[] rgbData;
  outputFile.close();
}

//*******************************************************************************************
// CONVOLUTION
//*******************************************************************************************

namespace PSkel{
 __parallel__ void stencilKernel(Array2D<float> &input,Array2D<float> &output,Mask2D<float> mask, Arguments args, size_t i, size_t j){
    //float accum = 0.0f;
    /*for(int n=0;n<mask.size;n++){
      accum += mask.get(n,input,i,j) * mask.getWeight(n);
    }
    output(i,j)= accum;
    */
    output(i,j) = input(i-2, j+2) * mask.getWeight(0) +
                  input(i-1, j+2) * mask.getWeight(1) +
                  input(i  , j+2) * mask.getWeight(2) +
                  input(i+1, j+2) * mask.getWeight(3) +
                  input(i+2, j+2) * mask.getWeight(4) +
                  input(i-2, j+1) * mask.getWeight(5) +
                  input(i-1, j+1) * mask.getWeight(6) +
                  input(i  , j+1) * mask.getWeight(7) +
                  input(i+1, j+1) * mask.getWeight(8) +
                  input(i+2, j+1) * mask.getWeight(9) +
                  input(i-2, j) * mask.getWeight(10) +
                  input(i-1, j) * mask.getWeight(11) +
                  input(i  , j) * mask.getWeight(12) +
                  input(i+1, j) * mask.getWeight(13) +
                  input(i+2, j) * mask.getWeight(14) +
                  input(i-2, j-1) * mask.getWeight(15) +
                  input(i-1, j-1) * mask.getWeight(16) +
                  input(i  , j-1) * mask.getWeight(17) +
                  input(i+1, j-1) * mask.getWeight(18) +
                  input(i+2, j-1) * mask.getWeight(19) +
                  input(i-2, j-2) * mask.getWeight(20) +
                  input(i-1, j-2) * mask.getWeight(21) +
                  input(i  , j-2) * mask.getWeight(22) +
                  input(i+1, j-2) * mask.getWeight(23) +
                  input(i+2, j-2) * mask.getWeight(24); 
  }
}//end namespace

//*******************************************************************************************
// MAIN
//*******************************************************************************************

int main(int argc, char **argv){  

  Arguments arg;
  arg.dummy = 1;
  Mask2D<float> mask(25);
  mask.set(0,-2,2,0.0); mask.set(1,-1,2,0.0); mask.set(2,0,2,0.0);  mask.set(3,1,2,0.0);  mask.set(4,2,2,0.0);
  mask.set(5,-2,1,0.0); mask.set(6,-1,1,0.0); mask.set(7,0,1,0.1);  mask.set(8,1,1,0.0);  mask.set(9,2,1,0.0);
  mask.set(10,-2,0,0.0);  mask.set(11,-1,0,0.1);  mask.set(12,0,0,0.2); mask.set(13,1,0,0.1); mask.set(14,2,0,0.0);
  mask.set(15,-2,-1,0.0); mask.set(16,-1,-1,0.0); mask.set(17,0,-1,0.1);  mask.set(18,1,-1,0.0);  mask.set(19,2,-1,0.0);
  mask.set(20,-2,-2,0.0); mask.set(21,-1,-2,0.0); mask.set(22,0,-2,0.0);  mask.set(23,1,-2,0.0);  mask.set(24,2,-2,0.0);
  
  int nb_tiles = atoi(argv[0]);
  int tilling_width = atoi(argv[1]);
  int tilling_height = atoi(argv[2]);
  int cluster_id = atoi(argv[3]);
  int nb_threads = atoi(argv[4]);
  int inner_iterations = atoi(argv[5]);
  int outter_iterations = atoi(argv[6]);
  int itMod = atoi(argv[7]);
  int nb_clusters = atoi(argv[8]);
  int width = atoi(argv[9]);
  int height = atoi(argv[10]);
  int nb_computated_tiles = atoi(argv[11]);

  int halo_value = mask.getRange() * inner_iterations;

  Array2D<float> partInput(tilling_width, tilling_height, halo_value);
  Array2D<float> output(tilling_width, tilling_height, halo_value);
  Stencil2D<Array2D<float>, Mask2D<float>, Arguments> stencil(partInput, output, mask, arg);

  stencil.runMPPA(cluster_id, nb_threads, nb_tiles, outter_iterations, inner_iterations, itMod, nb_clusters, width, height, nb_computated_tiles);

  return 0;
}

