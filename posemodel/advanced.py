from nbox import Operator
url = "https://i0.wp.com/post.healthline.com/wp-content/uploads/2020/01/Runner-training-on-running-track-1296x728-header-1296x728.jpg?w=1155&h=1528"

mediapipe = Operator.from_serving("https://api.nimblebox.ai/cdlmonrl/", "<token>")
out = mediapipe.predict_url(url)
print(out)
