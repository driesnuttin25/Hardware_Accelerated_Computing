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
    outputPrefix[strlen(imageFileName) - 4] = '\0'();

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









To accommodate the changes in the main.cu file in your notebook, you need to modify how you call the CUDA program so that it no longer expects the output prefix argument. Here are the specific changes needed:

Update the Notebook Code
Original Code to Run the CUDA Program:
python
Copy code
# Run the CUDA program on each image
for image in images:
    input_path = os.path.join(input_dir, image)
    output_prefix = os.path.splitext(image)[0]
    subprocess.run(['./main', input_path, output_prefix])
Change to:
python
Copy code
# Run the CUDA program on each image
for image in images:
    input_path = os.path.join(input_dir, image)
    subprocess.run(['./main', input_path])
Updated Notebook Code
Here is the complete updated section of the notebook that includes the changes to compile and run the CUDA program:

python
Copy code
import os
import subprocess

# Create directories
os.makedirs('input', exist_ok=True)
os.makedirs('output', exist_ok=True)

# Download images into the input directory
!wget -O input/darth.png https://raw.githubusercontent.com/driesnuttin25/Hardware_Accelerated_Computing/main/convolution/input/darth.png

# Compile the CUDA program
!nvcc main.cu -o main

# Get all images in the input directory
input_dir = 'input'
output_dir = 'output'
images = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

# Run the CUDA program on each image
for image in images:
    input_path = os.path.join(input_dir, image)
    subprocess.run(['./main', input_path])
Update the profiling and detailed analysis sections:
python
Copy code
import os
import subprocess

# Download the image
!wget -O input/darth.png https://raw.githubusercontent.com/driesnuttin25/Hardware_Accelerated_Computing/main/convolution/input/darth.png

# Compile the CUDA program
subprocess.run(['nvcc', 'main.cu', '-o', 'main'], check=True)

# Define input and output directories
input_dir = 'input'
output_dir = 'output'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set the image file name
image = 'darth.png'
input_path = os.path.join(input_dir, image)

# Use Nsight Systems to profile the application and generate a .qdstrm file
subprocess.run(['nsys', 'profile', '-o', 'profile_output', './main', input_path], check=True)
python
Copy code
import os
import subprocess
import time

# Function to print and run commands with timing and error handling
def run_command(command, description):
    print(f"Running: {description}")
    start_time = time.time()
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        end_time = time.time()
        print(f"Completed: {description} in {end_time - start_time:.2f} seconds")
        print("Standard Output:\n", result.stdout)
        print("Standard Error:\n", result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        print(f"Error: {description} failed after {end_time - start_time:.2f} seconds")
        print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}")
        print(f"Standard Output:\n{e.stdout}")
        print(f"Standard Error:\n{e.stderr}")
        return e

# Ensure input and output directories exist
input_dir = 'input'
output_dir = 'output'
if not os.path.exists(input_dir):
    os.makedirs(input_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Download the image
print("Downloading the image...")
!wget -O input/darth.png https://raw.githubusercontent.com/driesnuttin25/Hardware_Accelerated_Computing/main/convolution/input/darth.png

# Compile the CUDA program
run_command(['nvcc', 'main.cu', '-o', 'main'], "Compiling the CUDA program")

# Set the image file name
image = 'darth.png'
input_path = os.path.join(input_dir, image)

# Run Nsight Compute for detailed kernel analysis with full metrics
run_command(['ncu', '--set', 'full', '--export', 'nsight_compute_report', './main', input_path], "Running Nsight Compute")
These changes will ensure that your notebook correctly compiles and runs the CUDA program without requiring an output prefix, and all outputs will be saved to the output folder based on the image file names.
