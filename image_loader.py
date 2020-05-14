import multiprocessing as mp
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
import ctypes
import warnings
import numpy as np
import os
import cv2
import queue

from image_parser import ImageParser

# directory = 'images/'
directory = 'images_old/'
# directory = 'large_images/'

class ImageLoader(mp.Process):


	def __init__(self, idx, worker_stats, 
		filename_queue, color_buffer_queue, *args, **kwargs):
		
		super().__init__(*args, **kwargs)
		self.idx = idx
		self.worker_stats = worker_stats
		self.parser = ImageParser(idx, worker_stats)
		self.jpeg_loader = TurboJPEG()
		self.num_images = 0
		self.file_size_sum = 0
		self.pixel_sum = 0
		self.filename_queue = filename_queue
		self.color_buffer_queue = color_buffer_queue
		self.is_running = mp.Value(ctypes.c_bool, True)

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
		# cl0 = clock()
		buf = in_file.read()
		bgr_array = self.read_image(filename, buf)
		if bgr_array is None:
			return
		assert bgr_array.dtype == np.uint8
		self.parser.add_image(bgr_array)
		# self.parser_queue.put(bgr_array)
		# self.image_parser.add_image(bgr_array)
		# cl1 = clock()
		self.file_size_sum += os.path.getsize(directory + filename)
		self.pixel_sum += bgr_array.size // 3
		# print(f"{filename} parsed")
		# t1 = time.time()
		# self.print_stats(i, t0, t1, cl0, cl1)

	def run(self):
		self.parser.compile()
		while self.is_running.value:
		# while True:
			try:
				image_data = self.filename_queue.get(True, 1)
			except queue.Empty:
				continue
			filename = f"{image_data['filename']}.{image_data['filetype']}"
			with open(directory + filename, 'rb') as in_file:
				self.load_single_image(filename, in_file)
			# time.sleep(1)
			# print(f"Completed {image_data}")
		self.parser.finalize_parser()
		self.color_buffer_queue.put(self.parser.col_buffer)
