import multiprocessing as mp
import numpy as np
import time
from time import perf_counter as clock
import ctypes
import pyopencl as cl
import warnings
import functools

class ImageParser:

	def __init__(self, idx, worker_stats):
		self.idx = idx
		self.worker_stats = worker_stats
		self.buffer_size = 1 * 256 * 1024**2
		self.buffer = np.zeros(self.buffer_size, dtype = np.uint8)
		self.color_buffer = np.zeros(256**3, dtype = np.uint32)
		self.offset = 0
		self.num_pixels = 0
		self.total_pixels = 0
		self.total_images = 0
		self.start_time = time.time()
		self.is_running = mp.Value(ctypes.c_bool, True)

	def compile(self):
		self.ctx = cl.create_some_context()

		self.queue = cl.CommandQueue(self.ctx, 
			properties=cl.command_queue_properties.PROFILING_ENABLE)
		self.make_kernel()
		self.setup_cl_buffers()

	def make_kernel(self):
		kernel = open("color_kernel.cl", "r").read()
		t0 = clock()
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category = UserWarning)
			self.prg = cl.Program(self.ctx, 
				kernel).build(["-cl-fast-relaxed-math"])
		t1 = clock()
		self.compilation_time = t1-t0

	def setup_cl_buffers(self):
		mf = cl.mem_flags
		self.buffer_g = cl.Buffer(self.ctx, 
			mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = self.buffer)
		self.color_buffer_g = cl.Buffer(self.ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = self.color_buffer)
		cl.enqueue_copy(self.queue, self.buffer_g, self.buffer)
		self.prg.calculate_fast(self.queue, (10,), None, 
			self.buffer_g, self.color_buffer_g)


	def add_image(self, image):
		num_pixels = functools.reduce(lambda a, b: a*b, image.shape)
		num_dims = len(image.shape)
		if num_pixels > self.buffer_size:
			return
		if self.offset + num_pixels > self.buffer_size:
			self.parse_and_empty()

		image.shape = (num_pixels,)
		self.buffer[self.offset:self.offset+num_pixels] = image
		self.offset += num_pixels
		self.num_pixels += num_pixels // 3
		self.total_images += 1

	def parse_and_empty(self):
		self.total_pixels += self.num_pixels
		self.worker_stats[self.idx] = self.total_pixels
		mp = self.total_pixels / 1e6
		self.queue.finish()
		cl.enqueue_copy(self.queue, self.buffer_g, self.buffer)
		event = self.prg.calculate_fast(self.queue, (self.num_pixels,), None, 
			self.buffer_g, self.color_buffer_g)
		self.offset = 0
		self.num_pixels = 0

	def finalize_parser(self):
		self.parse_and_empty()
		self.fetch_color_buffer()

	def fetch_color_buffer(self):
		cl.enqueue_copy(self.queue, self.color_buffer, self.color_buffer_g)
		self.color_buffer.shape = (256, 256, 256)

	@property
	def col_buffer(self):
		return self.color_buffer

	def run(self):
		while self.is_running.value or self.parser_queue.qsize() > 0:
			try:
				image = self.parser_queue.get(True, 1)
			except queue.Empty:
				continue
			self.add_image(image)
		print("parser ended", self.parser_queue.qsize())
		self.finalize_parser()
