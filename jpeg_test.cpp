#include "turbojpeg.h"
#include <iostream>
#include <string.h>
#include <errno.h>
#include <fstream>
#include <vector>
#include <string>
#include <stdio.h>
#include <dirent.h>
#include <sys/time.h>
#include <omp.h>
#include <filesystem>

namespace fs = std::filesystem;

double get_wall_time() {
	struct timeval time;
	if (gettimeofday(&time,NULL)){
		return 0;
	}
	return (double)time.tv_sec + (double)time.tv_usec * 1e-6;
}


int compress() {
	unsigned char *srcBuf; //passed in as a param containing pixel data in RGB pixel interleaved format
	tjhandle handle = tjInitCompress();

	if(handle == NULL)
	{
		const char *err = (const char *) tjGetErrorStr();
		std::cerr << "TJ Error: " << err << " UNABLE TO INIT TJ Compressor Object\n";
		return -1;
	}
	int jpegQual =92;
	int width = 128;
	int height = 128;
	int nbands = 3;
	int flags = 0;
	unsigned char* jpegBuf = NULL;
	int pitch = width * nbands;
	int pixelFormat = TJPF_GRAY;
	int jpegSubsamp = TJSAMP_GRAY;
	if(nbands == 3)
	{
		pixelFormat = TJPF_RGB;
		jpegSubsamp = TJSAMP_411;
	}
	unsigned long jpegSize = 0;

	srcBuf = new unsigned char[width * height * nbands];
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			srcBuf[(j * width + i) * nbands + 0] = (i) % 256;
			srcBuf[(j * width + i) * nbands + 1] = (j) % 256;
			srcBuf[(j * width + i) * nbands + 2] = (j + i) % 256;
		}
	}

	int tj_stat = tjCompress2( handle, srcBuf, width, pitch, height,
		pixelFormat, &(jpegBuf), &jpegSize, jpegSubsamp, jpegQual, flags);
	if(tj_stat != 0)
	{
		const char *err = (const char *) tjGetErrorStr();
		std::cerr << "TurboJPEG Error: " << err << " UNABLE TO COMPRESS JPEG IMAGE\n";
		tjDestroy(handle);
		handle = NULL;
		return -1;
	}

	FILE *file = fopen("out.jpg", "wb");
	if (!file) {
		std::cerr << "Could not open JPEG file: " << strerror(errno);
		return -1;
	}
	if (fwrite(jpegBuf, jpegSize, 1, file) < 1) {
		std::cerr << "Could not write JPEG file: " << strerror(errno);
		return -1;
	}
	fclose(file);

	//write out the compress date to the image file
	//cleanup
	int tjstat = tjDestroy(handle); //should deallocate data buffer
	handle = 0;
	return 0;
}

std::vector<unsigned char> getInputChars(std::string filename, int &size) {

	std::ifstream in(filename);

	std::vector<unsigned char> contents((std::istreambuf_iterator<char>(in)), {});

	size = contents.size();
	return contents;

}

int decompress(tjhandle &_jpegDecompressor, unsigned char* _compressedImage, 
	int _jpegSize, std::vector<unsigned char> &buf) {
	
	// int _jpegSize;
	int width, height, jpegSubsamp, jpegColorspace;
	unsigned char* buffer;  
	// std::cout << "decompressing header of size " << _jpegSize << std::endl;
	int success = tjDecompressHeader3(_jpegDecompressor, _compressedImage, 
		_jpegSize, &width, &height, &jpegSubsamp, &jpegColorspace);

	// std::cout << "header decompressed: " << success << std::endl;
	// std::cout << tjGetErrorStr2(_jpegDecompressor) << std::endl;	

	// std::cout << width << "   " << height << "   " << _jpegSize << std::endl;

	buf.resize(width*height*3);
	// buffer = new unsigned char[width*height*3];
	// buffer = reinterpret_cast<unsigned char*>(buf.data());

	// tjDecompress2(_jpegDecompressor, _compressedImage, _jpegSize, buffer, 
	tjDecompress2(_jpegDecompressor, _compressedImage, _jpegSize, &buf[0], 
		width, 0, height, TJPF_RGB, TJFLAG_FASTDCT);
	// printf("buf size: %lu\n", buf.size());


	return 0;
}

std::vector<std::string> get_filenames(int argc, const char**argv) {
	std::vector<std::string> ret;
	std::string s;
	struct dirent *entry = nullptr;
	DIR *dp = nullptr;

	dp = opendir(argc > 1 ? argv[1] : "images");
	if (dp != nullptr) {
		while ((entry = readdir(dp)))
			printf ("%s\n", entry->d_name);
			s = entry->d_name;
			ret.push_back(s);
			printf("%lu\n", ret.size());
	}

	closedir(dp);
	return ret;
}

void get_average(std::vector<unsigned char> &buf, double &r, double &g, double &b) {
	int n = buf.size();
	// printf("size: %lu\n", n);
	for (int i = 0; i < n; i += 3) {
		r += buf[i];
		g += buf[i+1];
		b += buf[i+2];
	}
	r /= n/3;
	g /= n/3;
	b /= n/3;
}

void add_colors(std::vector<unsigned char> &buf, std::vector<int> &color) {
	int n = buf.size();
	for (int i = 0; i < n; i += 3) {
		int idx = 256 * (256 * buf[i] + buf[i+1]) + buf[i+2];
		color[idx]++;
	}
}

int check_image(std::string filename, std::vector<unsigned char> &image, 
	std::vector<unsigned char> buf, unsigned char* image_array,
	tjhandle &decompressor, std::vector<int> &colors) {

	int jpeg_size;
	image = getInputChars(filename, jpeg_size);
	image_array = reinterpret_cast<unsigned char*>(image.data());
	decompress(decompressor, image_array, jpeg_size, buf);
	add_colors(buf, colors);
	return jpeg_size;
}

int main(int argc, const char **argv)
{
	std::vector<std::string> filenames;
    for(auto& p: std::filesystem::directory_iterator("images")) {
    	filenames.push_back(p.path());
        // std::cout << p.path() << '\n';
    }

    std::cout << filenames.size() << '\n';
    int n = 2;
	if (argc > 1) {
		n = atoi(argv[1]);
	}
	omp_set_num_threads(n);
	std::vector<tjhandle> decompressors(n);
	std::vector<std::vector<int>> colors(8, std::vector<int>(256*256*256));
	for (int i = 0; i < n; i++) {
		decompressors[i] = tjInitDecompress();
	}
	std::vector<std::vector<unsigned char>> images(n);
	std::vector<std::vector<unsigned char>> bufs(n);
	std::vector<unsigned char*> image_arrays(n);
	double t0 = get_wall_time();
	int iters = filenames.size();
	int total_size = 0;
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < iters; i++) {
		int tid = omp_get_thread_num();
		int size = check_image(filenames[i], images[tid], bufs[tid], image_arrays[tid],
			decompressors[tid], colors[tid]);
		#pragma omp atomic
			total_size += size;
	}
	for (int i = 0; i < n; i++) {
		tjDestroy(decompressors[i]);
	}
	double t1 = get_wall_time();
	printf("%.3f MB/s\n", 1e-6 * total_size / (t1-t0));
}