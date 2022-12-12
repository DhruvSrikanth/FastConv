# Environment variables
compiler = g++-11

# Input arguments
image_path = data/input/example.png
kernel_choice = edge
output_path = data/output/example.png

# Compile the following
compile: ./conv2d.cpp
	$(compiler) ./conv2d.cpp -o ./conv2d -O3 -ffast-math -march=native -mtune=native -lm

# Execute the following
run: ./conv2d
	./conv2d $(image_path) $(kernel_choice) $(output_path)

# Clean the following
clean_runtime:
	@rm ray_tracing

clean_output:
	@rm -r data/output/*