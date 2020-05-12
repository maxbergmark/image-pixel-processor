import cv2
import numpy as np
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
import time
from time import perf_counter as clock
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2
import colorsys
import functools
import warnings
import multiprocessing
import pymysql.cursors
import queue
from db_credentials import DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE
import ctypes
import pyopencl as cl

# directory = 'images/'
directory = 'images_old/'
# directory = 'large_images/'

class ImageParser(multiprocessing.Process):


	def __init__(self, parser_queue, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.buffer_size = 2 * 256 * 1024**2
		self.buffer = np.zeros(self.buffer_size, dtype = np.uint8)
		self.color_buffer = np.zeros(256**3, dtype = np.uint32)
		self.parser_queue = parser_queue
		self.offset = 0
		self.num_pixels = 0
		self.total_pixels = 0
		self.total_images = 0
		self.start_time = time.time()
		self.is_running = multiprocessing.Value(ctypes.c_bool, True)
		# self.device = cl.get_platforms()[0].get_devices()
		# self.ctx = cl.Context(self.device)
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
		# print("adding image")
		# time.sleep(0.1)
		# return
		num_pixels = functools.reduce(lambda a, b: a*b, image.shape)
		num_dims = len(image.shape)
		# print(num_pixels, image.shape)
		if self.offset + num_pixels > self.buffer_size:
			# self.index_offsets[self.num_images] = self.offset
			self.parse_and_empty()

		image.shape = (num_pixels,)
		self.buffer[self.offset:self.offset+num_pixels] = image
		self.offset += num_pixels
		self.num_pixels += num_pixels // 3
		self.total_images += 1

	def parse_and_empty(self):
		self.total_pixels += self.num_pixels
		mp = self.total_pixels / 1e6
		t1 = time.time()
		print(f"parsing: {self.parser_queue.qsize():3d}\t"
			f"{self.total_images:6d}\t{mp:8.2f}MP\t"
			f"{mp / (t1-self.start_time):7.2f}MP/s")
		# print("parsing images,", self.parser_queue.qsize(), self.total_pixels / 1024**2)
		self.queue.finish()
		# print(self.buffer.shape, self.buffer_size)
		cl.enqueue_copy(self.queue, self.buffer_g, self.buffer)
		# print("parsed")

		# print(self.prg)
		event = self.prg.calculate_fast(self.queue, (self.num_pixels,), None, 
			self.buffer_g, self.color_buffer_g)
		self.offset = 0
		self.num_pixels = 0

	def finalize_parser(self):
		print("finalize")
		self.parse_and_empty()
		self._color_buffer = self.fetch_color_buffer()
		# ImageParser.is_running = False
		# t1 = time.time()
		# print((
			# f"Time passed: {t1-t0:.1f}s\t"
			# f"Total file size: {self.file_size_sum / 1024**3:.2f}GB\t"
			# f"Color sum: {self.color_buffer.sum()}"
		# ))

	def fetch_color_buffer(self):
		cl.enqueue_copy(self.queue, self.color_buffer, self.color_buffer_g)
		self.color_buffer.shape = (256, 256, 256)
		return self.color_buffer

	def run(self):
		while self.is_running.value or self.parser_queue.qsize() > 0:
			# print(self.is_running, self.parser_queue.qsize())
			try:
				image = self.parser_queue.get(True, 1)
			except queue.Empty:
				# print("empty")
				continue
			# print(image.shape)
			self.add_image(image)
		print("parser ended", self.parser_queue.qsize())
		self.finalize_parser()

class ImageLoader(multiprocessing.Process):


	def __init__(self, filename_queue, parser_queue, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.parser_queue = parser_queue
		self.jpeg_loader = TurboJPEG()
		self.num_images = 0
		self.file_size_sum = 0
		self.pixel_sum = 0
		self.filename_queue = filename_queue
		self.is_running = multiprocessing.Value(ctypes.c_bool, True)

	def read_png(self, buf):
		x = np.frombuffer(buf, dtype = np.uint8)
		img_np = cv2.imdecode(x, cv2.IMREAD_UNCHANGED)
		if img_np is not None:
			if img_np.dtype == np.uint16 and img_np.max() > 255:
				img_np = (img_np // 256).astype(np.uint8)
		return img_np

	def read_jpeg(self, buf):
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			return self.jpeg_loader.decode(buf)

	def read_image(self, filename, buf):

		if filename.endswith(".png"):
			bgr_array = self.read_png(buf)
		else:
			try:
				bgr_array = self.read_jpeg(buf)
			except OSError:
				bgr_array = None
		if bgr_array is not None:
			if len(bgr_array.shape) > 2 and bgr_array.shape[2] == 4:
				# return None
				# print("need to realign memory")
				bgr_array = np.ascontiguousarray(bgr_array[:,:,:3])
			if len(bgr_array.shape) == 2:
				new_array = np.zeros(bgr_array.shape + (3,), dtype = np.uint8)
				for i in range(3):
					new_array[:,:,i] = bgr_array
				bgr_array = new_array
				# print(bgr_array.shape)
		return bgr_array

	def print_stats(self, i, t0, t1, cl0, cl1):
		mp = self.pixel_sum / 1024**2
		mb = self.file_size_sum / 1024**2
		mp_per_second = mp / (t1 - t0)
		mb_per_second = mb / (t1 - t0)
		print(f"\r{i:4d}", end = "\t", flush = True)
		print(f"{mp_per_second:8.1f}MP/s", end = "\t", flush = True)
		print(f"{mb_per_second:.2f}MB/s", end = "\t")
		print(f"({(cl1-cl0) * 1e3:6.1f}ms) ({mp:7.1f}MP)", end = "")

	def load_single_image(self, filename, in_file):
		cl0 = clock()
		buf = in_file.read()
		bgr_array = self.read_image(filename, buf)
		if bgr_array is None:
			return
		assert bgr_array.dtype == np.uint8
		self.parser_queue.put(bgr_array)
		# self.image_parser.add_image(bgr_array)
		cl1 = clock()
		self.file_size_sum += os.path.getsize(directory + filename)
		self.pixel_sum += bgr_array.size // 3
		# print(f"{filename} parsed")
		# t1 = time.time()
		# self.print_stats(i, t0, t1, cl0, cl1)


	def load_all_images(self):
		t0 = time.time()

		# for i, filename in enumerate(os.listdir(directory)):
			# with open(directory + filename, 'rb') as in_file:
				
		print()
		self.finalize_parser(t0)

	def run(self):
		while self.is_running.value:
		# while True:
			if self.parser_queue.qsize() > 100:
				# print(self.parser_queue.qsize())
				time.sleep(.1)
				continue
			try:
				image_data = self.filename_queue.get(True, 1)
			except queue.Empty:
				continue
			filename = f"{image_data['filename']}.{image_data['filetype']}"
			with open(directory + filename, 'rb') as in_file:
				self.load_single_image(filename, in_file)
			# time.sleep(1)
			# print(f"Completed {image_data}")
		print("worker finished", self.filename_queue.qsize(), self.parser_queue.qsize())

	@property
	def color_buffer(self):
		return self._color_buffer

class WorkHandler:

	is_running = True

	def __init__(self, num_workers, num_parsers):
		self.filename_queue = multiprocessing.Queue()
		self.parser_queue = multiprocessing.Queue()
		self.num_workers = num_workers
		self.num_parsers = num_parsers
		self.current_session_max_id = 0
		self.min_size = 100
		self.batch_size = 200
		self.connection = pymysql.connect(
			host = DB_HOST,
			user = DB_USER,
			password = DB_PASSWORD,
			db = DB_DATABASE,
			charset = 'utf8mb4',
			cursorclass = pymysql.cursors.DictCursor
		)

	@staticmethod
	def interrupt():
		ImageDownloader.interrupt_all_workers()
		WorkHandler.is_running = False
		name = multiprocessing.current_process().name
		if name == "MainProcess":
			print("Interrupt received, exiting")

	def start_workers(self):
		self.image_parsers = []
		for i in range(self.num_parsers):
			parser = ImageParser(self.parser_queue)
			self.image_parsers.append(parser)
			# time.sleep(1)
		# self.image_parsers = [ImageParser(self.parser_queue)
			# for i in range(self.num_parsers)]
		self.workers = [ImageLoader(self.filename_queue, self.parser_queue) 
			for i in range(self.num_workers)]
		for w in self.workers:
			w.start()
		for p in self.image_parsers:
			p.start()

	def fill_queue_sql(self):
		query = (f"SELECT id, filename, filetype FROM image_metadata "
				f"WHERE is_downloaded = 1 "
				f"AND id > {self.current_session_max_id} "
				f"ORDER BY id "
				f"limit {self.batch_size}")

		with self.connection.cursor() as cursor:
			cursor.execute(query)
			result = cursor.fetchall()
			if len(result) == 0:
				WorkHandler.is_running = False
				print("all images loaded")
				return
			for r in result:
				self.filename_queue.put(r)
			self.current_session_max_id = result[-1]["id"]

	def fill_queue(self):
		for filename_full in os.listdir("images_old/"):
			# print(filename_full)
			file_info = filename_full.split(".")
			filename, filetype = ".".join(file_info[:-1]), file_info[-1]
			r = {"filename": filename, "filetype": filetype}
			self.filename_queue.put(r)
		WorkHandler.is_running = False

	def manage_work(self):
		self.fill_queue()
		self.start_workers()
		while WorkHandler.is_running:
			if self.filename_queue.qsize() < self.min_size:
				self.fill_queue()
			time.sleep(.01)

		self.finalize_work()

	def finalize_work(self):
		while self.filename_queue.qsize() > 0:
			time.sleep(.01)
		for w in self.workers:
			w.is_running.value = False
		for p in self.image_parsers:
			p.is_running.value = False
		print("joining processes", self.filename_queue.qsize(), self.parser_queue.qsize())
		for w in self.workers:
			w.join()
		print("workers joined")
		for p in self.image_parsers:
			p.join()
		print("parser joined")

def show_counters(counters):
	max_c = counters.max()
	counters = counters.astype(np.float64) / max_c

	i = 0
	while True:
		frame_resize = cv2.resize(counters[:,:,i % 256], 
					dsize = (1024, 1024))
		# cv2.imshow('Colors', counters[:,:,i % 256])
		cv2.imshow('Colors', frame_resize**.2)
		i += 1
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break

def visualize_counters(counters):
	counters.shape = (256, 256, 256)
	hues = np.zeros(255, dtype = np.int64)
	for r in range(256):
		for g in range(256):
			for b in range(256):
				hsv = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
				hue = int(hsv[0] * 255)
				hues[hue] += counters[b, g, r]

	plt.plot(hues)
	plt.show()

if __name__ == "__main__":

	print(cl.get_platforms())
	print(cl.get_platforms()[0].get_devices())
	print(cl.get_platforms()[1].get_devices())

	work_handler = WorkHandler(2, 2)
	work_handler.manage_work()
	# image_loader = ImageLoader()
	# image_loader.load_all_images()

	# show_counters(image_loader.color_buffer)