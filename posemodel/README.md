# Mediapipe Model

In this example we are going to deploy a (MediaPipe)[https://mediapipe.dev/] model for landmark detection. There are many ways to call a deployed a CV model:
- directly transfering uint8 array: as you would think this creates the heaviest message, this is the worst. Do it just to realise how bad it is.
- transferring the bytes of the image: this is an industry standard approach where the client would send in a base64 encoded image bytes and the server would reconstruct the image, this is a good case when your server is in an unsafe environment
- sending in a URL and server would fetch it: this is good when your server is in a safe environment and you know what are the actual contents of the URL

**Note on opencv**: `mediapipe` has a dependency on `opencv` and installing `opencv` is a bit tricky since it depends directly on the system packages. So we the trick the system by installing it right from inside our script even before `mediapipe` is imported, this may seem like a hack but it is **99% solution that works 99% of times**.

## Serve

The class is defined in `model.py` file and to serve this model run:
```
nbx serve upload model:MediaPipeModel 'mediapipe_model'
```

The way `nbox.Operator` works is that it would take all the functions that you have in a class and create an endpoint against it, in this case:

- `predict` would be served at `method_predict_rest/`, this takes in a raw array and returns predictions
- `predict_b64` would be served at `method_predict_b64_rest/`, this takes in a base64 encoded image
- `predict_url` would be served at `method_predict_url_rest/`, this takes in a URL

The developer is free from writing API endpoints, managing the complexity of on-wire protocols, they simply write functions that can take in any input (for REST it needs to be JSON serialisable).

## Use

The model is now deployed on an API endpoint that looks like this: `https://api.nimblebox.ai/cdlmonrl/`, you can go to the Deploy → 'mediapipe_model' → Settings and get your access key, it would look like this: `nbxdeploy_AZqcVWuVm0pC4k567EaUjOCOulZiQ3YdLEQJNnrR`. The file `predict.py` contains more detailed tests for the API endpoint. Here's from my run:

```
Time taken for array (avg. 10 calls): 9.3824s
Time taken for b64 (avg. 20 calls): 1.2814s
Time taken for url (avg. 50 calls): 0.4145s
```

## Advanced

The `nbox.Operator` is designed to wrap any arbitrary python class or function to become part of a distributed compute fabric. When you have deployed a model on NBX-Deploy you can connect directly via an `Operator` with `.from_serving` classmethod like:

```
mediapipe = Operator.from_serving("https://api.nimblebox.ai/cdlmonrl/", "<token>")
out = mediapipe.predict_url(url)
```

To test it run file:
```
python3 advanced.py
```
