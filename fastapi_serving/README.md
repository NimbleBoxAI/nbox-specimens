# Deploy any FastAPI app

With NimbleBox Serving you can run any FastAPI app without changing a single line of code. `server.py` contains a server to list and update items. Note that `serving_type = "fastapi_v2"` will override `/` and `/metadata` endpoints.

<pre><code>nbx serve upload server:app --id '<serving_id>' <b>--serving_type="fastapi_v2"</b></code></pre>

You can make a simple cURL requests like this:

```bash
curl -k -H 'NBX-KEY: <your-key>' https://api.nimblebox.ai/<deployment_id>/items/foo
```
