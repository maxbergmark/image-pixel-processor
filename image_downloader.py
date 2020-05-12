# from multiprocessing import multiprocessing.Process, Queue
import multiprocessing
import queue
import time
import signal
import sys
from db_credentials import DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE
import pymysql.cursors
import urllib.request
from urllib.error import HTTPError

IMAGE_DIRECTORY = "images/"

def sigint_handler(signal, frame):
	WorkHandler.interrupt()

signal.signal(signal.SIGINT, sigint_handler)



class ImageDownloader(multiprocessing.Process):

	workers = []

	def __init__(self, queue, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.queue = queue
		self.is_running = True
		ImageDownloader.workers.append(self)
		self.connection = pymysql.connect(
			host = DB_HOST,
			user = DB_USER,
			password = DB_PASSWORD,
			db = DB_DATABASE,
			charset = 'utf8mb4',
			cursorclass = pymysql.cursors.DictCursor
		)

	@staticmethod
	def interrupt_all_workers():
		for w in ImageDownloader.workers:
			w.cancel_run()

	def get_image_url(self, image_data):
		query = f"SELECT url FROM image_metadata WHERE id = {image_data['id']}"
		with self.connection.cursor() as cursor:
			cursor.execute(query)
			result = cursor.fetchone()
		return result["url"]

	def mark_as_saved(self, image_data):
		if "filetype" in image_data:
			query = (f"UPDATE image_metadata SET filename = '{image_data['filename']}', "
					f"filetype = '{image_data['filetype']}', "
					f"is_downloaded = true WHERE id = {image_data['id']}")
		else:
			query = (f"UPDATE image_metadata SET filename = '{image_data['filename']}', "
					f"is_downloaded = true WHERE id = {image_data['id']}")

		print(query)
		with self.connection.cursor() as cursor:
			cursor.execute(query)
			self.connection.commit()


	def download_and_save(self, image_data):
		try:
			url = image_data["url"]
			filename = image_data["post_id"]
			image_data["filename"] = filename
			if "filetype" in image_data:
				filename += f".{image_data['filetype']}"
			urllib.request.urlretrieve(url, IMAGE_DIRECTORY + filename)
			self.mark_as_saved(image_data)
		except HTTPError as e:
			print("Exception:", e)
			pass

	def download_image(self, image_data):
		url = self.get_image_url(image_data)
		image_data["url"] = url
		if url.endswith(".jpg") or url.endswith(".png"):
			image_data["filetype"] = url[-3:]
			self.download_and_save(image_data)
		elif url.startswith("https://i.reddituploads.com"):
			self.download_and_save(image_data)
		elif url.startswith("https://imgur.com") or url.startswith("http://imgur.com"):
			pass

	def run(self):
		while self.is_running and not self.queue.empty():
			try:
				image_data = self.queue.get(True, 1)
			except queue.Empty:
				break
			print(f"Doing {image_data}")
			self.download_image(image_data)
			# time.sleep(1)
			# print(f"Completed {image_data}")

	def cancel_run(self):
		self.is_running = False

class WorkHandler:

	is_running = True

	def __init__(self, num_workers):
		self.work_items = multiprocessing.Queue()
		self.num_workers = num_workers
		self.current_session_max_id = 0
		self.min_size = 10
		self.batch_size = 100
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
		self.workers = [ImageDownloader(self.work_items) 
			for i in range(self.num_workers)]
		for w in self.workers:
			w.start()

	def fill_queue(self):
		query = (f"SELECT id, post_id FROM image_metadata "
				f"WHERE NOT is_downloaded "
				f"AND id > {self.current_session_max_id} "
				f"ORDER BY id "
				f"limit {self.batch_size}")

		# for i in range(self.current_session_max_id, self.current_session_max_id + 10):
			# self.work_items.put(i)
		# self.current_session_max_id += 10
		with self.connection.cursor() as cursor:
			cursor.execute(query)
			result = cursor.fetchall()
			for r in result:
				self.work_items.put(r)
			self.current_session_max_id = result[-1]["id"]



	def manage_work(self):
		self.fill_queue()
		self.start_workers()
		while WorkHandler.is_running:
			if self.work_items.qsize() < self.min_size:
				self.fill_queue()
			time.sleep(1)

		self.finalize_work()



	def finalize_work(self):
		for w in self.workers:
			w.join()



work_handler = WorkHandler(1)
work_handler.manage_work()
