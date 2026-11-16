import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "input.txt")

# The tests use Tiny Shakespeare. Download it once if the data folder is empty (for example on a fresh clone).
if not os.path.exists(DATA):
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", DATA)
