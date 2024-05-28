// Welcome to this probably very badly written code to some but hey it works :)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void applyConvolution(unsigned char* image, unsigned char* output, int width, int height, int channels, float *kernel);
__global__ void applyMaxPooling(unsigned char* image, unsigned char* output, int width, int height, int channels);
__global__ void applyAveragePooling(unsigned char* image, unsigned char* output, int width, int height, int channels);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        return 1;
    }

    // Load the image
    int width, height, channels;
    unsigned char* img = stbi_load(argv[1], &width, &height, &channels, 0);
    if (img == NULL) {
        printf("Error in loading the image\n");
        return -1;
    }

    const char* imageFileName = strrchr(argv[1], '/');
    if (imageFileName == NULL) {
        imageFileName = argv[1];
    } else {
        imageFileName++;
    }
    char outputPrefix[256];
    strncpy(outputPrefix, imageFileName, strlen(imageFileName) - 4);
    outputPrefix[strlen(imageFileName) - 4] = '\0';

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Define convolution kernel
    float kernel[3][3] = {{1, 0, -1}, {1, 0, -1}, {1, 0, -1}};
    float* d_kernel;
    cudaMalloc(&d_kernel, 3 * 3 * sizeof(float));
    cudaMemcpy(d_kernel, kernel, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for image on device
    unsigned char* d_image;
    cudaMalloc(&d_image, width * height * channels * sizeof(unsigned char));
    cudaMemcpy(d_image, img, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Create CUDA streams
    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    // Allocate memory for convolution output and run convolution kernel
    unsigned char* convOutput = (unsigned char*)malloc(width * height * channels * sizeof(unsigned char));
    unsigned char* d_convOutput;
    cudaMalloc(&d_convOutput, width * height * channels * sizeof(unsigned char));
    applyConvolution<<<gridSize, blockSize, 0, stream1>>>(d_image, d_convOutput, width, height, channels, d_kernel);
    cudaMemcpyAsync(convOutput, d_convOutput, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream1);

    // Allocate memory for max pooling output and run max pooling kernel
    unsigned char* maxPoolOutput = (unsigned char*)malloc((width / 2) * (height / 2) * channels);
    unsigned char* d_maxPoolOutput;
    cudaMalloc(&d_maxPoolOutput, (width / 2) * (height / 2) * channels);
    applyMaxPooling<<<gridSize, blockSize, 0, stream2>>>(d_image, d_maxPoolOutput, width, height, channels);
    cudaMemcpyAsync(maxPoolOutput, d_maxPoolOutput, (width / 2) * (height / 2) * channels, cudaMemcpyDeviceToHost, stream2);

    // Allocate memory for average pooling output and run average pooling kernel
    unsigned char* avgPoolOutput = (unsigned char*)malloc((width / 2) * (height / 2) * channels);
    unsigned char* d_avgPoolOutput;
    cudaMalloc(&d_avgPoolOutput, (width / 2) * (height / 2) * channels);
    applyAveragePooling<<<gridSize, blockSize, 0, stream3>>>(d_image, d_avgPoolOutput, width, height, channels);
    cudaMemcpyAsync(avgPoolOutput, d_avgPoolOutput, (width / 2) * (height / 2) * channels, cudaMemcpyDeviceToHost, stream3);

    // Wait for all streams to complete
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);

    // Save output images
    char convOutputPath[256];
    snprintf(convOutputPath, sizeof(convOutputPath), "output/%s_conv_output.png", outputPrefix);
    stbi_write_png(convOutputPath, width, height, channels, convOutput, width * channels);
    printf("Finished Convolution for %s\n", outputPrefix);

    char maxPoolOutputPath[256];
    snprintf(maxPoolOutputPath, sizeof(maxPoolOutputPath), "output/%s_max_pool_output.png", outputPrefix);
    stbi_write_png(maxPoolOutputPath, width / 2, height / 2, channels, maxPoolOutput, (width / 2) * channels);
    printf("Finished Max pooling for %s\n", outputPrefix);

    char avgPoolOutputPath[256];
    snprintf(avgPoolOutputPath, sizeof(avgPoolOutputPath), "output/%s_avg_pool_output.png", outputPrefix);
    stbi_write_png(avgPoolOutputPath, width / 2, height / 2, channels, avgPoolOutput, (width / 2) * channels);
    printf("Finished Average pooling for %s\n", outputPrefix);

    // Cleanup
    stbi_image_free(img);
    free(convOutput);
    free(maxPoolOutput);
    free(avgPoolOutput);

    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_convOutput);
    cudaFree(d_maxPoolOutput);
    cudaFree(d_avgPoolOutput);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    return 0;
}

// Kernel for convolution operation
__global__ void applyConvolution(unsigned char* image, unsigned char* output, int width, int height, int channels, float *kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return; // Ensure we don't go out of bounds

    int edge = 1; // Kernel size is 3x3, so edge is 1
    float sum[3] = {0.0, 0.0, 0.0};

    // Apply convolution filter
    for (int ky = -edge; ky <= edge; ky++) {
        for (int kx = -edge; kx <= edge; kx++) {
            int ix = x + kx;
            int iy = y + ky;
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                for (int ch = 0; ch < channels; ch++) {
                    if (ch < 3) { // Apply convolution only to RGB channels
                        sum[ch] += kernel[(ky + edge) * 3 + (kx + edge)] * image[(iy * width + ix) * channels + ch];
                    }
                }
            }
        }
    }
    // Save the result back to the output image
    for (int ch = 0; ch < channels; ch++) {
        if (ch < 3) {
            int val = (int)sum[ch];
            output[(y * width + x) * channels + ch] = (unsigned char)(val > 255 ? 255 : (val < 0 ? 0 : val));
        } else {
            output[(y * width + x) * channels + ch] = image[(y * width + x) * channels + ch];
        }
    }
}

// Kernel for max pooling operation
__global__ void applyMaxPooling(unsigned char* image, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return; // Ensure we don't go out of bounds

    int outputWidth = width / 2;
    int outputHeight = height / 2;

    // Apply max pooling filter
    for (int y = 0; y < outputHeight; y++) {
        for (int x = 0; x < outputWidth; x++) {
            for (int ch = 0; ch < channels; ch++) {
                unsigned char maxVal = 0;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int iy = y * 2 + dy;
                        int ix = x * 2 + dx;
                        unsigned char val = image[(iy * width + ix) * channels + ch];
                        if (val > maxVal) maxVal = val;
                    }
                }
                output[(y * outputWidth + x) * channels + ch] = maxVal;
            }
        }
    }
}

// Kernel for average pooling operation
__global__ void applyAveragePooling(unsigned char* image, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return; // Ensure we don't go out of bounds

    int outputWidth = width / 2;
    int outputHeight = height / 2;

    // Apply average pooling filter
    for (int y = 0; y < outputHeight; y++) {
        for (int x = 0; x < outputWidth; x++) {
            for (int ch = 0; ch < channels; ch++) {
                unsigned int sum = 0;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int iy = y * 2 + dy;
                        int ix = x * 2 + dx;
                        sum += image[(iy * width + ix) * channels + ch];
                    }
                }
                output[(y * outputWidth + x) * channels + ch] = sum / 4;
            }
        }
    }
}





// Welcome to this probably very badly written code to some but hey it works :)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void applyConvolution(unsigned char* image, unsigned char* output, int width, int height, int channels, float *kernel);
__global__ void applyMaxPooling(unsigned char* image, unsigned char* output, int width, int height, int channels);
__global__ void applyAveragePooling(unsigned char* image, unsigned char* output, int width, int height, int channels);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        return 1;
    }

    // Load the image
    int width, height, channels;
    unsigned char* img = stbi_load(argv[1], &width, &height, &channels, 0);
    if (img == NULL) {
        printf("Error in loading the image\n");
        return -1;
    }

    const char* imageFileName = strrchr(argv[1], '/');
    if (imageFileName == NULL) {
        imageFileName = argv[1];
    } else {
        imageFileName++;
    }
    char outputPrefix[256];
    strncpy(outputPrefix, imageFileName, strlen(imageFileName) - 4);
    outputPrefix[strlen(imageFileName) - 4] = '\0';

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Define convolution kernel
    float kernel[3][3] = {{1, 0, -1}, {1, 0, -1}, {1, 0, -1}};
    float* d_kernel;
    cudaMalloc(&d_kernel, 3 * 3 * sizeof(float));
    cudaMemcpy(d_kernel, kernel, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for image on device
    unsigned char* d_image;
    cudaMalloc(&d_image, width * height * channels * sizeof(unsigned char));
    cudaMemcpy(d_image, img, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Create CUDA streams
    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    // Allocate memory for convolution output and run convolution kernel
    unsigned char* convOutput = (unsigned char*)malloc(width * height * channels * sizeof(unsigned char));
    unsigned char* d_convOutput;
    cudaMalloc(&d_convOutput, width * height * channels * sizeof(unsigned char));
    applyConvolution<<<gridSize, blockSize, 0, stream1>>>(d_image, d_convOutput, width, height, channels, d_kernel);
    cudaMemcpyAsync(convOutput, d_convOutput, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream1);

    // Allocate memory for max pooling output and run max pooling kernel
    unsigned char* maxPoolOutput = (unsigned char*)malloc((width / 2) * (height / 2) * channels);
    unsigned char* d_maxPoolOutput;
    cudaMalloc(&d_maxPoolOutput, (width / 2) * (height / 2) * channels);
    applyMaxPooling<<<gridSize, blockSize, 0, stream2>>>(d_image, d_maxPoolOutput, width, height, channels);
    cudaMemcpyAsync(maxPoolOutput, d_maxPoolOutput, (width / 2) * (height / 2) * channels, cudaMemcpyDeviceToHost, stream2);

    // Allocate memory for average pooling output and run average pooling kernel
    unsigned char* avgPoolOutput = (unsigned char*)malloc((width / 2) * (height / 2) * channels);
    unsigned char* d_avgPoolOutput;
    cudaMalloc(&d_avgPoolOutput, (width / 2) * (height / 2) * channels);
    applyAveragePooling<<<gridSize, blockSize, 0, stream3>>>(d_image, d_avgPoolOutput, width, height, channels);
    cudaMemcpyAsync(avgPoolOutput, d_avgPoolOutput, (width / 2) * (height / 2) * channels, cudaMemcpyDeviceToHost, stream3);

    // Wait for all streams to complete
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);

    // Save output images
    char convOutputPath[256];
    snprintf(convOutputPath, sizeof(convOutputPath), "output/%s_conv_output.png", outputPrefix);
    stbi_write_png(convOutputPath, width, height, channels, convOutput, width * channels);
    printf("Finished Convolution for %s\n", outputPrefix);

    char maxPoolOutputPath[256];
    snprintf(maxPoolOutputPath, sizeof(maxPoolOutputPath), "output/%s_max_pool_output.png", outputPrefix);
    stbi_write_png(maxPoolOutputPath, width / 2, height / 2, channels, maxPoolOutput, (width / 2) * channels);
    printf("Finished Max pooling for %s\n", outputPrefix);

    char avgPoolOutputPath[256];
    snprintf(avgPoolOutputPath, sizeof(avgPoolOutputPath), "output/%s_avg_pool_output.png", outputPrefix);
    stbi_write_png(avgPoolOutputPath, width / 2, height / 2, channels, avgPoolOutput, (width / 2) * channels);
    printf("Finished Average pooling for %s\n", outputPrefix);

    // Cleanup
    stbi_image_free(img);
    free(convOutput);
    free(maxPoolOutput);
    free(avgPoolOutput);

    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_convOutput);
    cudaFree(d_maxPoolOutput);
    cudaFree(d_avgPoolOutput);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    return 0;
}

// Kernel for convolution operation
__global__ void applyConvolution(unsigned char* image, unsigned char* output, int width, int height, int channels, float *kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return; // Ensure we don't go out of bounds

    int edge = 1; // Kernel size is 3x3, so edge is 1
    float sum[3] = {0.0, 0.0, 0.0};

    // Apply convolution filter
    for (int ky = -edge; ky <= edge; ky++) {
        for (int kx = -edge; kx <= edge; kx++) {
            int ix = x + kx;
            int iy = y + ky;
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                for (int ch = 0; ch < channels; ch++) {
                    if (ch < 3) { // Apply convolution only to RGB channels
                        sum[ch] += kernel[(ky + edge) * 3 + (kx + edge)] * image[(iy * width + ix) * channels + ch];
                    }
                }
            }
        }
    }
    // Save the result back to the output image
    for (int ch = 0; ch < channels; ch++) {
        if (ch < 3) {
            int val = (int)sum[ch];
            output[(y * width + x) * channels + ch] = (unsigned char)(val > 255 ? 255 : (val < 0 ? 0 : val));
        } else {
            output[(y * width + x) * channels + ch] = image[(y * width + x) * channels + ch];
        }
    }
}

// Kernel for max pooling operation
__global__ void applyMaxPooling(unsigned char* image, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width / 2 || y >= height / 2) return; // Ensure we don't go out of bounds in the output

    for (int ch = 0; ch < channels; ch++) {
        unsigned char maxVal = 0;
        for (int dy = 0; dy < 2; dy++) {
            for (int dx = 0; dx < 2; dx++) {
                int ix = x * 2 + dx;
                int iy = y * 2 + dy;
                if (ix < width && iy < height) { // Ensure we don't go out of bounds in the input
                    unsigned char val = image[(iy * width + ix) * channels + ch];
                    if (val > maxVal) maxVal = val;
                }
            }
        }
        output[(y * (width / 2) + x) * channels + ch] = maxVal;
    }
}

// Kernel for average pooling operation
__global__ void applyAveragePooling(unsigned char* image, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width / 2 || y >= height / 2) return; // Ensure we don't go out of bounds in the output

    for (int ch = 0; ch < channels; ch++) {
        unsigned int sum = 0;
        int count = 0;
        for (int dy = 0; dy < 2; dy++) {
            for (int dx = 0; dx < 2; dx++) {
                int ix = x * 2 + dx;
                int iy = y * 2 + dy;
                if (ix < width && iy < height) { // Ensure we don't go out of bounds in the input
                    sum += image[(iy * width + ix) * channels + ch];
                    count++;
                }
            }
        }
        output[(y * (width / 2) + x) * channels + ch] = sum / count;
    }
}
