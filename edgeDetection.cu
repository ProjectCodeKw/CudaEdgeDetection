
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib> // For system()
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>
#include <math.h>
#include <cstdlib> // For exit()
using namespace std;

#define BLOCK_SIZE 16 // 2D block size

// Function to read PPM file (Grayscale format P5)
vector<unsigned char> readPPM(const string &filename, int &width, int &height)
{
    ifstream file(filename, ios::binary);
    if (!file)
    {
        cerr << "Error opening PPM file\n";
        exit(1);
    }

    string magic;
    file >> magic; // Read P5
    if (magic != "P5")
    {
        cerr << "Invalid PPM format\n";
        exit(1);
    }

    // Read width, height, and max color value
    file >> width >> height;
    int maxVal;
    file >> maxVal;
    file.ignore(1); // Consume newline before pixel data

    // Read pixel data (grayscale)
    vector<unsigned char> pixelData(width * height);
    file.read(reinterpret_cast<char *>(pixelData.data()), pixelData.size());

    return pixelData;
}

void CPU_EDGEDETECTION(unsigned char *a, unsigned char *c, int width, int height)
{
    for (int y = 0; y < height - 1; y++)
    {
        for (int x = 0; x < width - 1; x++)
        {
            float f_xy1 = a[y * width + (x + 1)];        // f(x, y+1)
            float f_x1y = a[(y + 1) * width + x];        // f(x+1, y)
            float f_x1y1 = a[(y + 1) * width + (x + 1)]; // f(x+1, y+1)
            float f_xy = a[y * width + x];               // f(x, y)
            float gx = f_xy1 - f_x1y;                    // (f(x,y+1) - f(x+1,y))
            float gy = f_x1y1 - f_xy;                    // (f(x+1,y+1) - f(x,y))
            c[y * width + x] = sqrtf(gx * gx + gy * gy);
        }
    }
}

__global__ void EDGEDETECTION(unsigned char *a, unsigned char *c, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // thread location
    int y = blockIdx.y * blockDim.y + threadIdx.y; // thread location

    if (x < width - 1 && y < height - 1)

    {
        float f_xy1 = a[y * width + (x + 1)];        // f(x, y+1)
        float f_x1y = a[(y + 1) * width + x];        // f(x+1, y)
        float f_x1y1 = a[(y + 1) * width + (x + 1)]; // f(x+1, y+1)
        float f_xy = a[y * width + x];               // f(x, y)

        float gx = f_xy1 - f_x1y; // (f(x,y+1) - f(x+1,y))
        float gy = f_x1y1 - f_xy; // (f(x+1,y+1) - f(x,y))

        c[y * width + x] = sqrtf(gx * gx + gy * gy);
    }
}

int main()
{

    // read the ppm file
    string path = "og_image.ppm";
    int width, height;
    vector<unsigned char> h_a = readPPM(path, width, height);
    size_t size = h_a.size();

    // allocatin memory for the sizes in the GPU
    unsigned char *d_a, *d_c;
    cudaMalloc(&d_a, size);           // input image
    cudaMalloc(&d_c, width * height); // output edge

    // copy to the GPU
    cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);                                       // Block dimensions in 2D
    dim3 gridSize((int)ceil(width / BLOCK_SIZE), (int)ceil(height / BLOCK_SIZE)); // width, height

    clock_t start_cpu = clock();
    CPU_EDGEDETECTION(h_a.data(), h_a.data(), width, height);
    clock_t end = clock();
    double cpu_time_used = ((double)(end - start_cpu)) / CLOCKS_PER_SEC;
    printf("CPU took: %f ms to execute, \n", cpu_time_used * 1000);

    cudaEvent_t start, stop; // for GPU timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    EDGEDETECTION<<<gridSize, blockSize>>>(d_a, d_c, width, height);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start, stop);
    printf("GPU time: %0.6f ms\n", milliseconds1);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    // Copy result back to host
    vector<unsigned char> h_c(width * height);
    cudaMemcpy(h_c.data(), d_c, width * height, cudaMemcpyDeviceToHost);

    // Save output to file (Binary PPM P5) and overwrite it if it exists
    ofstream outFile("edge_result.ppm", ios::binary | ios::trunc);
    outFile << "P5\n"
            << width << " " << height << "\n255\n";
    outFile.write(reinterpret_cast<char *>(h_c.data()), h_c.size());
    outFile.close();

    printf("SPEED UP: %0.2f\n", (cpu_time_used * 1000) / milliseconds1);

    // Free memory
    cudaFree(d_a);
    cudaFree(d_c);

    exit(0);
}
