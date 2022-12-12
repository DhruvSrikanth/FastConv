# Environment variables
cuda_version=11.1
export CUDA_HOME=/usr/local/cuda-$(cuda_version)
export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(CUDA_HOME)/lib64
export PATH=$(PATH):/usr/local/cuda/bin

C_Compiler = g++-12
CUDA_Compiler = nvcc

# Input arguments
image_path = data/input/example.png
kernel_choice = edge
output_path = data/output/example.png

# CUDA code arguments
n_threads_per_block = 1024

# Serial run recipes
c_sim: c_compile c_run clean_runtime

# Compile the following
c_compile: ./conv2d.cpp
	$(C_Compiler) ./conv2d.cpp -o ./conv2d.out -O3 -ffast-math -mtune=native -lm -w

# Execute the following
c_run: ./conv2d.out
	./conv2d.out $(image_path) $(kernel_choice) $(output_path)

# CUDA run recipes
cuda_sim: cuda_compile cuda_run clean_runtime

# Compile the following
cuda_compile: ./conv2d.cu
	$(CUDA_Compiler) ./conv2d.cu -o ./conv2d.out -O3 -use_fast_math -extra-device-vectorization -lm

# Execute the following
cuda_run: ./conv2d.out
	./conv2d.out $(image_path) $(kernel_choice) $(output_path) $(n_threads_per_block)

# Clean the following
clean_runtime:
	@rm ./conv2d.out

clean_output:
	@rm -r data/output/*