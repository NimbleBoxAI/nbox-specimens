from fastapi import FastAPI, Response

app = FastAPI()

ITEMS = {
  "foo": "The Foo Wrestlers",
  "bar": "The Bar Fighters",
  "wysiwyg": "What You See Is What You Get",
  "memes": "Memes are the best",
  "fastapi": "FastAPI is a server",
}

@app.get("/")
def read_root():
  return {
    "message": "Hello World!"
  }

@app.get("/items/{item_id}")
def read_item(item_id: str, resp: Response):
  if item_id in ITEMS:
    return {
      "item_id": item_id,
      "item_name": ITEMS[item_id]
    }
  else:
    resp.status_code = 404
    return {
      "message": "Item not found"
    }

@app.post("/update_item_details")
def update_item_details(item_id: str, item_name: str, resp: Response):
  if item_id in ITEMS:
    ITEMS[item_id] = item_name
    return {
      "item_id": item_id,
      "item_name": item_name
    }
  else:
    resp.status_code = 404
    return {
      "message": "Item not found"
    }
