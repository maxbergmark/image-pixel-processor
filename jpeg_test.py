import cv2
import numpy as np
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
import time
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2
import colorsys

jpeg = TurboJPEG()
directory = 'images/'
# directory = 'large_images/'

def read_png(buf):
	# nparr = np.fromstring(buf, np.uint8)
	x = np.frombuffer(buf, dtype='uint8')
	img_np = cv2.imdecode(x, cv2.IMREAD_UNCHANGED)
	if img_np is not None:
		if img_np.dtype == np.uint16 and img_np.max() > 255:
			img_np = img_np // 256
		# img_np = img_np.astype(np.int32)
	return img_np

def read_jpeg(buf):
	return jpeg.decode(buf)
	# return jpeg.decode(buf).astype(np.int32)

def read_image(filename, buf):

	if filename.endswith(".png"):
		bgr_array = read_png(buf)
	else:
		try:
			bgr_array = read_jpeg(buf)
		except OSError:
			bgr_array = None

	return bgr_array
	# if bgr_array is None:
		# return None

	# if bgr_array.dtype != np.uint8:
		# print("\n\n\nError", bgr_array.dtype)

	# if bgr_array.max() > 255 or bgr_array.min() < 0:
		# print("Error")
		# print(bgr_array.min(), bgr_array.max(), bgr_array.shape)
		# print(bgr_array)
	"""
	if len(bgr_array.shape) == 2:
		bgr_new = np.zeros((bgr_array.shape[0], bgr_array.shape[1], 3), dtype = np.int32)
		bgr_new[:,:,0] = bgr_array
		bgr_new[:,:,1] = bgr_array
		bgr_new[:,:,2] = bgr_array
		bgr_array = bgr_new
		# print(bgr_array.shape)

	# raw_array = np.zeros(bgr_array.shape[:2], dtype = np.int32)
	# raw_array |= bgr_array[:,:,0] << 16
	# raw_array += bgr_array[:,:,1] << 8
	# raw_array += bgr_array[:,:,2] << 0
	raw_array = 256 * (256 * bgr_array[:,:,0] + bgr_array[:,:,1]) + bgr_array[:,:,2]
	raw_array.shape = (raw_array.shape[0] * raw_array.shape[1],)
	return raw_array
	"""

def increase_counters(buf, counters):
	counters[raw_array] += 1

def show_counters(counters):
	max_c = counters.max()
	print(max_c)
	counters = counters.astype(np.float64) / max_c
	counters.shape = (256, 256, 256)


	i = 0
	while True:
		frame_resize = cv2.resize(counters[:,:,i % 256], 
					dsize = (1024, 1024))
		# cv2.imshow('Colors', counters[:,:,i % 256])
		cv2.imshow('Colors', frame_resize**.5)
		i += 1
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break

counters = np.zeros(256**3, dtype =np.int32)

file_size_sum = 0
t0 = time.time() 
for i, filename in enumerate(os.listdir(directory)):
	# if filename != "1lnpzawt2ln31.png":
		# continue
	# if i > 200:
		# break
	with open(directory + filename, 'rb') as in_file:
		cl0 = time.perf_counter()
		buf = in_file.read()
		print(i, filename, end = "\t", flush = True)
		raw_array = read_image(filename, buf)
		# if raw_array is None:
			# continue
		# increase_counters(buf, counters)
		cl1 = time.perf_counter()
		file_size_sum += os.path.getsize(directory + filename)
		t1 = time.time()
		print("%.3fMB/s (%.3fms)" % (file_size_sum / 1024**2 / (t1 - t0), 1e3 * (cl1 - cl0)))
t1 = time.time()
print(t1-t0, file_size_sum)


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