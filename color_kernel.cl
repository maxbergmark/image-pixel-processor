// #include <stdio.h>



kernel void calculate(
	__global unsigned char *buffer, 
	volatile __global unsigned int *color_buffer,
	__global unsigned int *image_dims,
	__global unsigned char *image_num_dims,
	__global unsigned int *index_offsets
) {
	int idx = get_global_id(0);
	unsigned char num_dims = image_num_dims[idx];
	__global unsigned int *dims = &image_dims[3*idx];
	unsigned int offset = index_offsets[idx];
	__global unsigned char *image = &buffer[offset];
	unsigned int num_values = index_offsets[idx+1] - offset;

	if (num_dims == 3 && dims[2] == 3) {
		printf("%2d\t%9d\n", idx, num_values / 3);
		for (int i = 0; i < num_values; i += 3) {
			int b = image[i+0];
			int g = image[i+1];
			int r = image[i+2];
			unsigned int color_value = 256 * (256 * r + g) + b;
			color_buffer[color_value]++;
			// atomic_inc(&color_buffer[color_value]);

		}
	} else {
		printf("skipping\n");
	}

	// printf("%2d %02d %02d\n", idx, image[3], buffer[offset+3]);

}

kernel void calculate_fast(
	__global unsigned char *buffer, 
	volatile __global unsigned int *color_buffer
) {

	int idx = get_global_id(0);
	int b = buffer[3 * idx + 0];
	int g = buffer[3 * idx + 1];
	int r = buffer[3 * idx + 2];
	unsigned int color_value = 256 * (256 * r + g) + b;
	// color_buffer[color_value]++;
	atomic_inc(&color_buffer[color_value]);
	
}