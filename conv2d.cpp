#include <iostream>
#include <math.h>
#include <fstream>

void conv2D(double *out, double *in, double *kernel, const int width, const int height, const int kernel_size) {
    // Compute convolution for each pixel
    for (int y := 0; y < height; y++) {
        for (int x := 0; x < width; x++) {
            frobenius_norm(in, kernel, width, height, kernel_size, x, y)
        }
    }
}

void frobenius_norm(double *out, double *in, double *kernel, const int width, const int height, const int kernel_size, const int x, const int y) {
    // Image Shift
    const int shift = kernel_size / 2;
    int y_shift = 0;
    int x_shift = 0;

    // Output
    double r_out = 0.0;
    double g_out = 0.0;
    double b_out = 0.0;
    double a_out = 0.0;

    // Index
    int img_index = 0;
    int kernel_index = 0;

    // Compute frobenius norm
    for (int j := 0; j < kernel_size; j++) {
        // Compute shift in y direction
        y_shift = y + j - shift;
        for (int i := 0; i < kernel_size; i++) {
            // Compute shift in x direction
            const int x_shift = x + i - shift;

            // Check if pixel is in image (if not, skip i.e. use 0 padding)
            if (x_shift < 0 || x_shift >= width || y_shift < 0 || y_shift >= height) {
                continue;
            }

            // Compute index
            img_index = (y_shift * width + x_shift) * 4;
            kernel_index = j * kernel_size + i;

            // Compute output
            r_out += in[img_index] * kernel[kernel_index];
            g_out += in[index + 1] * kernel[kernel_index + 1];
            b_out += in[index + 2] * kernel[kernel_index + 2];

            // Compute alpha channel
            if (j == shift_y && i == shift_x) {
                a_out = in[index + 3];
            }
        }
    }

    // Write output
    const int out_index = (y * width + x) * 4;

    out[out_index] = r_out;
    out[out_index + 1] = g_out;
    out[out_index + 2] = b_out;
    out[out_index + 3] = a_out;
}

double* determine_kernel(string kernel_choice) {
    double *kernel;
    switch (kernel_choice) {
        case "blur":
            kernel = new double[9] {
                1/9.0, 1/9.0, 1/9.0,
                1/9.0, 1/9.0, 1/9.0,
                1/9.0, 1/9.0, 1/9.0
            };
        case "sharpen":
            kernel = new double[9] {
                0, -1, 0,
                -1, 5, -1,
                0, -1, 0
            };
        case "edge":
            kernel = new double[9] {
                -1, -1, -1,
                -1, 8, -1,
                -1, -1, -1
            };
        case "emboss":
            kernel = new double[9] {
                -2, -1, 0,
                -1, 1, 1,
                0, 1, 2
            };
        default:
            kernel = new double[9] {
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0
            };
    }
    return kernel;
}


void process_image(string image_path, string kernel_choice, string output_path) {
    // Read image
    int width, height, channels;
    unsigned char *image = stbi_load(image_path.c_str(), &width, &height, &channels, 4);

    // Determine kernel
    double *kernel = determine_kernel(kernel_choice);

    // Allocate memory for output image
    double *out = new double[width * height * 4];

    // Start timer
    auto start = chrono::high_resolution_clock::now();

    // Compute convolution
    conv2D(out, image, kernel, width, height, 3);

    // Stop timer
    auto stop = chrono::high_resolution_clock::now();

    // Compute time
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    cout << "Time taken by function: " << duration.count() << " microseconds" << endl;

    // Write output image
    stbi_write_png(output_path, width, height, 4, out, width * 4);

    // Free memory
    delete[] out;
    delete[] kernel;
    stbi_image_free(image);
}

int main(int argc, char** argv) {
    // Get the image path, kernel choice and output path
    string image_path = argv[1];
    string kernel_choice = argv[2];
    string output_path = argv[3];


    // Process image
    // process_image(image_path, kernel_choice, output_path);

    return 0;
}