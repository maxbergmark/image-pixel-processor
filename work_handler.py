import multiprocessing as mp
import ctypes
import pymysql.cursors
import os
import time
import numpy as np

from db_credentials import DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE
from image_loader import ImageLoader


class WorkHandler:

	is_running = True

	def __init__(self, num_workers):
		self.filename_queue = mp.Queue()
		self.color_buffer_queue = mp.Queue()
		self.worker_stats = mp.Array(
			ctypes.c_uint64, [0 for _ in range(num_workers)])
		self.num_workers = num_workers
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
	"""
	@staticmethod
	def interrupt():
		ImageDownloader.interrupt_all_workers()
		WorkHandler.is_running = False
		name = mp.current_process().name
		if name == "MainProcess":
			print("Interrupt received, exiting")
	"""
	def start_workers(self):
		self.workers = [ImageLoader(i, self.worker_stats,  
			self.filename_queue, self.color_buffer_queue) 
			for i in range(self.num_workers)]
		for w in self.workers:
			w.start()

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
			for _ in range(3):
				self.filename_queue.put(r)
		WorkHandler.is_running = False

	def manage_work(self):
		self.fill_queue()
		self.start_workers()
		self.start_time = time.time()
		while WorkHandler.is_running:
			if self.filename_queue.qsize() < self.min_size:
				self.fill_queue()
			time.sleep(.01)
			print(self.worker_stats)

		self.finalize_work()

	def finalize_work(self):
		while self.filename_queue.qsize() > 0:
			time.sleep(.1)
			pixel_sum = sum(self.worker_stats[:]) / 1e6
			t0, t1 = self.start_time, time.time()
			print(f"\rTotal pixels: {pixel_sum:10.2f}MP\t"
				f"Speed: {pixel_sum / (t1-t0):10.2f}MP/s", end = "")
		print()
		for w in self.workers:
			w.is_running.value = False

		while self.color_buffer_queue.qsize() < self.num_workers:
			time.sleep(.01)

		self.color_buffer = np.zeros((256, 256, 256), dtype = np.uint32)
		while self.color_buffer_queue.qsize() > 0:
			q_buf = self.color_buffer_queue.get()
			self.color_buffer += q_buf

		for w in self.workers:
			w.join()

		print(self.color_buffer.sum())
