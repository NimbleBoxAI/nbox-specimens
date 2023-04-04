# Deploy any FastAPI app with NimbleBox

Welcome to the NimbleBox Serving tutorial! With NimbleBox Serving, you can run any FastAPI app without changing a single line of code. The server.py file contains a server to list and update items. Note that setting `serving_type = "fastapi_v2"` will override the `/` and `/metadata` endpoints (this behaviour will become default in the future).

1. To deploy your FastAPI app, follow these steps. This command will upload your app to NimbleBox Serving and configure it for FastAPI. You can adjust the serving_type parameter as per your app requirements.

```bash
nbx serve upload server:app --id '<serving_id>' \
  --serving_type="fastapi_v2"
```

2. You can make a simple cURL request like this to retrieve the foo item from your app:

```bash
curl -k -H 'NBX-KEY: <your-key>' https://api.nimblebox.ai/<deployment_id>/items/foo
```

NimbleBox Serving makes it easy to deploy your FastAPI apps. With our platform, you can easily manage your deployments and scale your apps to meet your needs.

Thank you for choosing NimbleBox for your machine learning needs!
