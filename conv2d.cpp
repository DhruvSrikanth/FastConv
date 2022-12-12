#include <iostream>
using namespace std;
#include <chrono>
#include <math.h>
#include <fstream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct Pixel {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

void initialize_image(Pixel **image, const int width, const int height) {
    // Initialize image
    for (int i = 0; i < width * height; i++) {
        (*image)[i].r = 0;
        (*image)[i].g = 0;
        (*image)[i].b = 0;
    }
}

void create_image(Pixel **image, string image_path, int *width, int *height, int *channels, const int byte_stride) {
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

void write_image(Pixel **out, string output_path, const int width, const int height, const int channels, const int byte_stride) {
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

uint16_t clamp(const double value) {
    if (value < 0) {
        return 0;
    } else if (value > 255) {
        return 255;
    } else {
        return (uint16_t) value;
    }
}

void frobenius_norm(Pixel **out, Pixel **in, double *kernel, const int width, const int height, const int kernel_size, const int x, const int y) {
    // Image Shift
    const int shift = kernel_size / 2;
    int y_shift = 0;
    int x_shift = 0;

    // Index
    int img_index = 0;
    int kernel_index = 0;
    const int out_index = (y * width + x);

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
            const int x_shift = x + i - shift;

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

void conv2D(Pixel **out, Pixel **in, double *kernel, const int width, const int height, const int kernel_size) {
    // Compute convolution for each pixel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            frobenius_norm(out, in, kernel, width, height, kernel_size, x, y);
        }
    }
}

double* determine_kernel(string kernel_choice) {
    double *kernel;
    if (kernel_choice == "blur") {
        kernel = new double[9] {
            1/9.0, 1/9.0, 1/9.0,
            1/9.0, 1/9.0, 1/9.0,
            1/9.0, 1/9.0, 1/9.0
        };
    } else if (kernel_choice == "sharpen") {
        kernel = new double[9] {
            0, -1, 0,
            -1, 5, -1,
            0, -1, 0
        };
    } else if (kernel_choice == "edge") {
        kernel = new double[9] {
            -1, -1, -1,
            -1, 8, -1,
            -1, -1, -1
        };
    } else if (kernel_choice == "emboss") {
        kernel = new double[9] {
            -2, -1, 0,
            -1, 1, 1,
            0, 1, 2
        };
    } else {
        kernel = new double[9] {
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0
        };
    }
    return kernel;
}

void process_image(string image_path, string kernel_choice, string output_path) {
    // Create image
    int width, height, channels;
    const int byte_stride = 3;
    Pixel *image;
    create_image(&image, image_path, &width, &height, &channels, byte_stride);

    // Determine kernel
    const int kernel_size = 3;
    double *kernel = determine_kernel(kernel_choice);

    // Allocate memory for output image and initialize
    Pixel *out = new Pixel[width * height];
    initialize_image(&out, width, height);

    // Start timer
    auto start = chrono::high_resolution_clock::now();

    // Compute convolution
    conv2D(&out, &image, kernel, width, height, kernel_size);

    // Stop timer
    auto stop = chrono::high_resolution_clock::now();

    // Compute time
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    cout << "Time taken: " << duration.count() * 1e-3 << " milliseconds" << endl;

    // Write output image
    write_image(&out, output_path, width, height, channels, byte_stride);
    

    // Free memory
    delete[] image;
    delete[] out;
    delete[] kernel;
}

int main(int argc, char** argv) {
    // Get the image path, kernel choice and output path
    string image_path = argv[1];
    string kernel_choice = argv[2];
    string output_path = argv[3];

    // Process image
    process_image(image_path, kernel_choice, output_path);

    return 0;
}