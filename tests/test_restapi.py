import requests
import io
from PIL import Image
import base64

def img_to_base64(img_path):
    """
    Convert image (RGB) to base64 string
    """
    s = ''
    with Image.open(img_path) as img:
        with io.BytesIO() as buffered:
            img.save(buffered, format="JPEG")
            s += base64.b64encode(buffered.getvalue()).decode("ascii")
    return f'"data:image/jpeg;base64,{s}"'

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "dataset/seg_pred/seg_pred/3.jpg"

# load the input image and construct the payload for the request
payload = img_to_base64(IMAGE_PATH)
# submit the request
r = requests.post(KERAS_REST_API_URL, data=payload, 
                  headers={'Content-Type': 'application/json', "Accept-Encoding": "gzip, deflate"}).json()
# ensure the request was successful
if 'result' in r.keys():
    print(f'{r["result"]}: {r["probability"]:.4}')

# otherwise, the request failed
else:
    print("Request failed")


import timeit
t = timeit.timeit(setup = f"import requests; payload='{payload}'; url='{KERAS_REST_API_URL}';", 
                  stmt = "requests.post(url, data=payload, " +
                  "headers={'Content-Type': 'application/json'}).json()", number=10)
print(t)

