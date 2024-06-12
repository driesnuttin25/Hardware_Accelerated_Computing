# Hardware Accelerated Computing

# CUDA Programming Guide

This guide provides an introduction to writing CUDA code to accelerate your applications. We will cover the basics of setting up a CUDA environment, writing and compiling CUDA code, and using CUDA features such as memory management and streams.

## Table of Contents
1. [Setting Up the Environment](#setting-up-the-environment)
2. [Writing CUDA Code](#writing-cuda-code)
3. [Compiling and Running CUDA Code](#compiling-and-running-cuda-code)
4. [Memory Management in CUDA](#memory-management-in-cuda)
5. [Using CUDA Streams](#using-cuda-streams)
6. [Example: Accelerating a For Loop](#example-accelerating-a-for-loop)

## Setting Up the Environment

To develop and run CUDA code, you need to have a system with an NVIDIA GPU and the CUDA Toolkit installed. In Google Colab, you can set up the environment using the following commands:

```bash
!wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
!mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
!apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
!add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
!apt-get update
!apt-get -y install cuda
```
## Writing CUDA Code
CUDA code consists of host code (running on the CPU) and device code (running on the GPU). Here is a simple structure for a CUDA program:

```cuda
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel function to be executed on the GPU
__global__ void kernelFunction() {
    // Kernel code
}

int main() {
    // Host code
    kernelFunction<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}
```
## Example: Accelerating a For Loop
Let's accelerate a simple for loop that adds two arrays using CUDA. First, we write the CUDA kernel function:

```cuda
__global__ void addArrays(int *a, int *b, int *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}
```
Next, we allocate memory and transfer data between the host and device:

```cuda
int size = 1000;
int *h_a, *h_b, *h_c; // Host arrays
int *d_a, *d_b, *d_c; // Device arrays

// Allocate host memory
h_a = (int*)malloc(size * sizeof(int));
h_b = (int*)malloc(size * sizeof(int));
h_c = (int*)malloc(size * sizeof(int));

// Initialize host arrays
for (int i = 0; i < size; i++) {
    h_a[i] = i;
    h_b[i] = i;
}

// Allocate device memory
cudaMalloc((void**)&d_a, size * sizeof(int));
cudaMalloc((void**)&d_b, size * sizeof(int));
cudaMalloc((void**)&d_c, size * sizeof(int));

// Copy data from host to device
cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);

// Launch the kernel
int threadsPerBlock = 256;
int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
addArrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

// Copy result back to host
cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

// Free device memory
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);

// Free host memory
free(h_a);
free(h_b);
free(h_c);
```
## Compiling and Running CUDA Code
To compile and run CUDA code, you can use the nvcc compiler. In Google Colab, use the following commands:

```bash
!nvcc -o add_arrays add_arrays.cu
!./add_arrays
```
## Memory Management in CUDA
CUDA provides functions for managing memory on the GPU:

- cudaMalloc: Allocates memory on the GPU.
- cudaFree: Frees allocated GPU memory.
- cudaMemcpy: Copies data between host and device.

### Example
```cuda
int *d_a;
int size = 1000 * sizeof(int);

// Allocate memory on the device
cudaMalloc((void**)&d_a, size);

// Free the allocated memory
cudaFree(d_a);
```
## Using CUDA Streams
CUDA streams allow for concurrent execution of multiple kernels and memory operations.

### Example
```cuda
cudaStream_t stream;
cudaStreamCreate(&stream);

int *d_a, *d_b;
int size = 1000 * sizeof(int);

cudaMalloc((void**)&d_a, size);
cudaMalloc((void**)&d_b, size);

int *h_a = (int*)malloc(size);
int *h_b = (int*)malloc(size);

cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream);
cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream);

addArrays<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_a, d_b, d_c, size);

cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream);

cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);

cudaFree(d_a);
cudaFree(d_b);
free(h_a);
free(h_b);
```
## Example: Accelerating a For Loop
Here is a complete example that accelerates a for loop using CUDA to add two arrays:

```cuda
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addArrays(int *a, int *b, int *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int size = 1000;
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;

    h_a = (int*)malloc(size * sizeof(int));
    h_b = (int*)malloc(size * sizeof(int));
    h_c = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }

    cudaMalloc((void**)&d_a, size * sizeof(int));
    cudaMalloc((void**)&d_b, size * sizeof(int));
    cudaMalloc((void**)&d_c, size * sizeof(int));

    cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    addArrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

    cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```
By following this guide, you should be able to set up your environment, write and compile CUDA code, manage memory, and use CUDA streams for concurrent execution.


# Image Convolution Accelerated on a GPU with CUDA

This project involves developing a CUDA-based application tailored for the NVIDIA Jetson Nano. The primary objective is to efficiently process a sequence of 10 images, which can be sourced from predetermined URLs. Attention should be given to the dimensions of the images to ensure compatibility with the application's processing capabilities.

### Objective
The core functionality of the application centers around performing 2D convolution operations on the images. This process is aimed at demonstrating the profound impact of GPU acceleration on image processing tasks, leveraging the computational power of CUDA on the Jetson Nano.

### Demonstration
The following image shows the working of convolution and how it can make an image more blurry: 
<img width="703" alt="image" src="https://github.com/driesnuttin25/Hardware_Accelerated_Computing/assets/114076101/0f7f36a8-432c-4cc9-8680-4f164d272bcb">


#### Image Prior to Convolution
![image](https://github.com/driesnuttin25/Hardware_Accelerated_Computing/assets/114076101/0b32f3e9-e6dd-4d3f-9e80-6912705b54b6)

#### Image Following Convolution
![image](https://github.com/driesnuttin25/Hardware_Accelerated_Computing/assets/114076101/be81b134-f6d6-4471-abf8-ea698d747190)

By incorporating CUDA into the processing pipeline, the application showcases a significant enhancement in performance, enabling real-time processing capabilities that are pivotal for a wide array of computational tasks.

