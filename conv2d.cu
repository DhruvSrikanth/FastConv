#include <iostream>
using namespace std;
#include <chrono>
#include <math.h>
#include <fstream>
#include<cuda.h>
#include<cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Maximum number of blocks in a device grid (for each dim)
#define MAX_BLOCKS 65535

// Min function
#define MIN(a,b) (((a)<(b))?(a):(b))

struct Pixel {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

__host__ void initialize_image(Pixel **image, const int width, const int height) {
    // Initialize image
    for (int i = 0; i < width * height; i++) {
        (*image)[i].r = 0;
        (*image)[i].g = 0;
        (*image)[i].b = 0;
    }
}

__host__ void create_image(Pixel **image, const string image_path, int *width, int *height, int *channels, const int byte_stride) {
    // Read image
    unsigned char *file = stbi_load(image_path.c_str(), width, height, channels, byte_stride);

    // Allocate memory for image
    *image = new Pixel[*width * *height];
    initialize_image(image, *width, *height);

    // Create image
    for (int i = 0; i < *width * *height; i++) {
        (*image)[i].r = (uint8_t) file[i * byte_stride];
        (*image)[i].g = (uint8_t) file[i * byte_stride + 1];
        (*image)[i].b = (uint8_t) file[i * byte_stride + 2];
    }

    // Free memory
    stbi_image_free(file);
}

__host__ void write_image(Pixel **out, const string output_path, const int width, const int height, const int channels, const int byte_stride) {
    // Allocate memory for output image
    unsigned char *file = new unsigned char[width * height * byte_stride];

    // Create output image
    for (int i = 0; i < width * height; i++) {
        file[i * byte_stride] = (*out)[i].r;
        file[i * byte_stride + 1] = (*out)[i].g;
        file[i * byte_stride + 2] = (*out)[i].b;
    }

    // Write output image
    stbi_write_png(output_path.c_str(), width, height, channels, file, width * byte_stride);

    // Free memory
    delete[] file;
}

__device__ uint16_t clamp(const double value) {
    if (value < 0) {
        return 0;
    } else if (value > 255) {
        return 255;
    } else {
        return (uint16_t) value;
    }
}

__device__ void frobenius_norm(Pixel **out, Pixel **in, const double *kernel, const int width, const int height, const int kernel_size, const int x, const int y) {
    // Image Shift
    const int shift = kernel_size / 2;
    int y_shift = 0;
    int x_shift = 0;

    // Cuda image, out and kernel index (host and device index are the same because we spawn one thread per pixel)
    int img_index = 0;
    int kernel_index = 0;
    const int out_index = y * width + x;

    // Output
    double out_r = 0.0;
    double out_g = 0.0;
    double out_b = 0.0;

    // Compute frobenius norm
    for (int j = 0; j < kernel_size; j++) {
        // Compute shift in y direction
        y_shift = y + j - shift;
        for (int i = 0; i < kernel_size; i++) {
            // Compute shift in x direction
            x_shift = x + i - shift;

            // Check if pixel is in image (if not, skip i.e. use 0 padding)
            if (x_shift < 0 || x_shift > width - 1 || y_shift < 0 || y_shift > height - 1) {
                continue;
            }

            // Compute index
            img_index = y_shift * width + x_shift;
            kernel_index = j * kernel_size + i;
            

            // Compute output
            out_r += (double) (*in)[img_index].r * kernel[kernel_index];
            out_g += (double) (*in)[img_index].g * kernel[kernel_index];
            out_b += (double) (*in)[img_index].b * kernel[kernel_index];
        }
    }

    // Clamp output
    (*out)[out_index].r = clamp(out_r);
    (*out)[out_index].g = clamp(out_g);
    (*out)[out_index].b = clamp(out_b);
}

__global__ void conv2D(Pixel **out, Pixel **in, const double *kernel, const int width, const int height, const int kernel_size) {
    // Compute convolution for each pixel
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    frobenius_norm(out, in, kernel, width, height, kernel_size, x, y);
}

__host__ const double* determine_kernel(const string kernel_choice) {
    const double *kernel;
    if (kernel_choice == "blur") {
        kernel = new const double[9] {
            1/9.0, 1/9.0, 1/9.0,
            1/9.0, 1/9.0, 1/9.0,
            1/9.0, 1/9.0, 1/9.0
        };
    } else if (kernel_choice == "sharpen") {
        kernel = new const double[9] {
            0, -1, 0,
            -1, 5, -1,
            0, -1, 0
        };
    } else if (kernel_choice == "edge") {
        kernel = new const double[9] {
            -1, -1, -1,
            -1, 8, -1,
            -1, -1, -1
        };
    } else if (kernel_choice == "emboss") {
        kernel = new const double[9] {
            -2, -1, 0,
            -1, 1, 1,
            0, 1, 2
        };
    } else {
        kernel = new const double[9] {
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0
        };
    }
    return kernel;
}

__host__ void process_image(const string image_path, const string kernel_choice, const string output_path, const int nthreads_per_block) {
    // Create image on host
    int width, height, channels;
    const int byte_stride = 3;
    Pixel *image;
    create_image(&image, image_path, &width, &height, &channels, byte_stride);

    // Create image on device
    int width_d, height_d, channels_d;
    const int byte_stride_d;
    Pixel *image_d;
    cudaMalloc(&width_d, sizeof(int));
    cudaMalloc(&height_d, sizeof(int));
    cudaMalloc(&channels_d, sizeof(int));
    cudaMalloc(&byte_stride_d, sizeof(int));
    cudaMalloc(image_d, width * height * byte_stride * sizeof(Pixel));

    // Copy image to device
    cudaMemcpy(&width_d, &width, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&height_d, &height, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&channels_d, &channels, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&byte_stride_d, &byte_stride, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(image_d, image, width * height * byte_stride * sizeof(Pixel), cudaMemcpyHostToDevice);

    // Determine kernel on host
    const int kernel_size = 3;
    const double *kernel = determine_kernel(kernel_choice);

    // Create kernel on device
    const int kernel_size_d;
    const double *kernel_d;
    cudaMalloc(&kernel_size_d, sizeof(int));
    cudaMalloc(kernel_d, kernel_size * kernel_size * sizeof(double));

    // Copy kernel to device
    cudaMemcpy(&kernel_size_d, &kernel_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_d, kernel, kernel_size * kernel_size * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate memory for output image on host and initialize
    Pixel *out = new Pixel[width * height];
    initialize_image(&out, width, height);

    // Allocate memory for output image on device
    Pixel *out_d;
    cudaMalloc(out_d, width * height * sizeof(Pixel));
    cudaMemcpy(out_d, out, width * height * sizeof(Pixel), cudaMemcpyHostToDevice);

    // Compute the number of blocks
    const int x_blocks = MIN((width/n_threads_per_block) + 1, MAX_BLOCKS);
    const int y_blocks = MIN((height/n_threads_per_block) + 1, MAX_BLOCKS);
    const dim3 block_size(n_threads_per_block, n_threads_per_block);
    const dim3 grid_size(x_blocks, y_blocks);

    // CUDA timer
    cudaEvent_t start_device, stop_device;  
    float time_device;

    // Create timers
    cudaEventCreate(&start_device);
    cudaEventCreate(&stop_device);

    // Start timer
    cudaEventRecord(start_device, 0);  

    // Compute convolution
    conv2D<<<grid_size, block_size>>>(out_d, image_d, kernel_d, width_d, height_d, kernel_size_d);

    // Stop timer
    cudaEventRecord(stop_device, 0);
    cudaEventSynchronize(stop_device);
    cudaEventElapsedTime(&time_device, start_device, stop_device);

    // Compute time
    cout << "Time: " << time_device << " ms" << endl;

    // Copy output image from device to host
    cudaMemcpy(out, out_d, width * height * sizeof(Pixel), cudaMemcpyDeviceToHost);

    // Write output image
    write_image(&out, output_path, width, height, channels, byte_stride);
    

    // Free memory on host
    delete[] image;
    delete[] out;
    delete[] kernel;

    // Free memory on device
    cudaFree(&width_d);
    cudaFree(&height_d);
    cudaFree(&channels_d);
    cudaFree(&byte_stride_d);
    cudaFree(&kernel_size_d);
    cudaFree(image_d);
    cudaFree(out_d);
    cudaFree(kernel_d);

}

__host__ int main(int argc, char** argv) {
    // Get the input args
    const string image_path = argv[1];
    const string kernel_choice = argv[2];
    const string output_path = argv[3];
    const int nthreads_per_block = atoi(argv[4]);

    // Process image
    process_image(image_path, kernel_choice, output_path, nthreads_per_block);

    return 0;
}