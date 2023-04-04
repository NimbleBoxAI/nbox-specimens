# Mediapipe Model

> For a simpler example, see the [fastAPI](../fastapi_serving/) example.

This is a guide on how to deploy a MediaPipe model for landmark detection using NimbleBox Deploy. The guide explains the different ways to call a deployed computer vision (CV) model and the considerations when deploying OpenCV as a dependency.

The main class is defined in the model.py file and to serve this model, you can run the following command:

```
nbx serve upload model:MediaPipeModel 'your_id_goes_here'
```

The `nbox.Operator` works by taking all the functions that are in a class and creating an endpoint for each of them. In this case, the `predict` function is served at `method_predict_rest/`, which takes a raw array and returns predictions. The `predict_b64` function is served at `method_predict_b64_rest/`, which takes a base64-encoded image, and the `predict_url` function is served at `method_predict_url_rest/`, which takes a URL.

The `nbox.Operator` is designed to wrap any arbitrary Python class or function to become part of a distributed [compute fabric](../compute_fabric/). Once the model is deployed on NimbleBox Deploy, you can connect directly via an `Operator` with the `.from_serving` class method as follows:

```
mediapipe = Operator.from_serving("https://api.nimblebox.ai/cdlmonrl/", "<token>")
out = mediapipe.predict_url(url)
```

To test it, run the advanced.py file.

To call a deployed CV model, there are three ways:

- Directly transferring a uint8 array: This creates the heaviest message and is the worst way to call a deployed CV model.
- Transferring the bytes of the image: This is an industry-standard approach where the client sends a base64-encoded image byte, and the server reconstructs the image. This is a good approach when the server is in an unsafe environment.
- Sending a URL, and the server would fetch it: This approach is good when the server is in a safe environment, and the contents of the URL are known.

The `mediapipe` dependency has a dependency on `opencv`, and installing `opencv` is a bit tricky because it depends directly on the system packages. To solve this, the system can be tricked by installing it inside the script even before `mediapipe` is imported. This approach may seem like a hack, but it is a 99% solution that works 99% of the time.

To deploy the model, run the following command:

```bash
nbx serve upload model:MediaPipeModel 'mediapipe_model'
```

To use the model, go to the "Deploy" → "mediapipe_model" → "Settings" and get the access key. The API endpoint looks like this: `https://api.nimblebox.ai/dfsdffe/` and the access key would look something like `nbxdeploy_AZqcVWuVm0pC4k567EaUjOCOulZiQ3YdLEQJNnrR`. The `predict.py` file contains more detailed tests for the API endpoint, including the time taken for array, b64, and URL calls. The file `predict.py` contains more detailed tests for the API endpoint. Here's from my run:

```
Time taken for array (avg. 10 calls): 9.3824s
Time taken for b64 (avg. 20 calls): 1.2814s
Time taken for url (avg. 50 calls): 0.4145s
```
