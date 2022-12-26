import time
import requests
import numpy as np
from PIL import Image
from base64 import b64encode
from io import BytesIO


# r = requests.post(
#   "https://api.nimblebox.ai/ag67bguk/18uh7jqv/rest_predict_url",
#   headers = {"NBX-KEY": "<token>"},
#   json = {
#     "url": "https://i0.wp.com/post.healthline.com/wp-content/uploads/2020/01/Runner-training-on-running-track-1296x728-header-1296x728.jpg?w=1155&h=1528"
#   }
# )
# r.raise_for_status()
# print(r.json())


r = requests.get(
  "https://i0.wp.com/post.healthline.com/wp-content/uploads/2020/01/Runner-training-on-running-track-1296x728-header-1296x728.jpg?w=1155&h=1528"
)
img = Image.open(BytesIO(r.content))
_shape = np.array(img).shape[:2]

times = []
n = 10
print("Testing predict_rest")
for _ in range(n):
  st = time.time()
  r = requests.post(
    "https://api.nimblebox.ai/<serving_id>/rest_predict",
    headers = {"NBX-KEY": "<token>"},
    json = {
      "image_array": np.array(img).tolist()
    }
  )
  r.raise_for_status()
  et = time.time()
  times.append(et - st)

_mt = np.mean(times)
print(f"Time taken for array (avg. {n} calls): {_mt:0.4f}s")
# print(r.json())


times = []
n = 20
print("Testing predict_b64")
for _ in range(n):
  st = time.time()
  r = requests.post(
    "https://api.nimblebox.ai/<serving_id>/rest_predict_b64",
    headers = {"NBX-KEY": "<token>"},
    json = {
      "image_b64": b64encode(img.tobytes()).decode("utf-8"),
      "shape": _shape
    }
  )
  r.raise_for_status()
  et = time.time()
  times.append(et - st)

_mt = np.mean(times)
print(f"Time taken for b64 (avg. {n} calls): {_mt:0.4f}s")
# print(r.json())

times = []
n = 50
print("Testing predict_url")
for _ in range(n):
  st = time.time()
  r = requests.post(
    "https://api.nimblebox.ai/ag67bguk/18uh7jqv/rest_predict_url",
    headers = {"NBX-KEY": "<token>"},
    json = {
      "url": "https://i0.wp.com/post.healthline.com/wp-content/uploads/2020/01/Runner-training-on-running-track-1296x728-header-1296x728.jpg?w=1155&h=1528"
    }
  )
  r.raise_for_status()
  et = time.time()
  times.append(et - st)

_mt = np.mean(times)
print(f"Time taken for url (avg. {n} calls): {_mt:0.4f}s")
