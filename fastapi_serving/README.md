# Deploy any FastAPI app

With NimbleBox Serving you can run any FastAPI app without changing a single line of code. `server.py` contains a server to list and update items.

<pre><code>nbx serve upload server:app --id '<serving_id>' <b>--serving_type="fastapi"</b></code></pre>

## APIs

All your APIs will be available with a simple `/x/` suffix. For example, the `items` API will be available at `/x/items/`.

<pre><code>https://api.nimblebox.ai/ago7kguk/j0wnnjmx/<b>x/items</b></code></pre>

You can make a simple cURL requests like this:

```bash
curl https://api.nimblebox.ai/agj792uk/j0wnnjmx/x/items/foo -H 'NBX-KEY: <your-key>'
```
