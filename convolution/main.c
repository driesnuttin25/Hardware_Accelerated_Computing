#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>

void applyConvolution(unsigned char* image, unsigned char* output, int width, int height, int channels, float kernel[3][3]) {
    int edge = 1; // Since kernel size is 3x3

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum[3] = {0.0, 0.0, 0.0}; // Sum for each channel

            for (int ky = -edge; ky <= edge; ky++) {
                for (int kx = -edge; kx <= edge; kx++) {
                    int ix = x + kx;
                    int iy = y + ky;
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        for (int ch = 0; ch < channels; ch++) {
                            if (ch < 3) { // Apply convolution only to RGB channels
                                sum[ch] += kernel[ky + edge][kx + edge] * image[(iy * width + ix) * channels + ch];
                            }
                        }
                    }
                }
            }
            for (int ch = 0; ch < channels; ch++) {
                if (ch < 3) {
                    int val = (int)sum[ch];
                    output[(y * width + x) * channels + ch] = (unsigned char)(val > 255 ? 255 : (val < 0 ? 0 : val));
                } else {
                    // Preserve the alpha channel if present
                    output[(y * width + x) * channels + ch] = 255;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    unsigned char* img = stbi_load(argv[1], &width, &height, &channels, 0);
    if (img == NULL) {
        printf("Error in loading the image\n");
        exit(1);
    }

    // Define your convolution kernel
    float kernel[3][3] = {
        {1, 0, -1},
        {1, 0, -1},
        {1, 0, -1}
    };

    unsigned char* outputImg = (unsigned char*)malloc(width * height * channels);
    applyConvolution(img, outputImg, width, height, channels, kernel);

    stbi_write_png("output.png", width, height, channels, outputImg, width * channels);

    stbi_image_free(img);
    free(outputImg);

    return 0;
}
