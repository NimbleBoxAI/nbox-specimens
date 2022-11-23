from fastapi import FastAPI

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
  return '''<html>Hello World</html>'''

@app.get("/items/{item_id}")
def read_item(item_id: str):
  if item_id in ITEMS:
    return f'''<html>Item ID: {item_id} - {ITEMS[item_id]}</html>'''
  else:
    return f'''<html>Item ID: {item_id} - Not Found</html>'''

@app.post("/update_item_details")
def update_item_details(item_id: str, item_name: str):
  if item_id in ITEMS:
    ITEMS[item_id] = item_name
    return f'''<html>Item ID: {item_id} - {ITEMS[item_id]}</html>'''
  else:
    return f'''<html>Item ID: {item_id} - Not Found</html>'''
