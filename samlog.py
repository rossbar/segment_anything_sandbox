import numpy as np
import matplotlib.pyplot as plt
from skimage import color, util
from pathlib import Path
from tqdm import tqdm
import urllib.request
import tifffile as tff
import time

from segment_anything import SamPredictor, sam_model_registry

SAM_WEIGHTS_URL = {
    "default": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

# Cache downloads
cache_dir = Path.home() / ".cache/sam"
cache_dir.mkdir(parents=True, exist_ok=True)
weight_path = cache_dir / SAM_WEIGHTS_URL["default"].split("/")[-1]

# Download model weights if necessary
if not weight_path.exists():
    print("Downloading model weights...")
    
    def prog_hook(t):
        last = [0]
        def update_to(b=1, bsize=1, tsize=None):
            t.total = tsize
            t.update((b - last[0]) * bsize)
            last[0] = b
        return update_to
        
    weight_url = SAM_WEIGHTS_URL["default"]
        
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urllib.request.urlretrieve(weight_url, weight_path, reporthook=prog_hook(t))

    print("Done")
    
# Set up model
device = "cpu"
model_type = "default"
model = sam_model_registry[model_type](weight_path)
model.to(device)
predictor = SamPredictor(model)

# Example img
img_url = (
    "https://github.com/vanvalenlab/intro-to-deepcell/blob/master/"
    "pretrained_models/resources/example_input_combined.tif?raw=true"
)
img_path = Path.cwd() / "example_input_combined.tif"
if not img_path.exists():
    urllib.request.urlretrieve(img_url, img_path)
img = tff.imread(img_path)

# Image pre-processing
img = img.sum(axis=0)  # Combine nuclear and membrane into single channel
img = color.gray2rgb(img)  # Convert to rgb TODO: is this necessary for model?
img -= img.min()  # fp from 0. to 1.
img /= img.max()
util.img_as_ubyte(img)
img = util.img_as_ubyte(img)  # Convert to uint8 (model requirement)

# Finish model setup
print("Loading image to model...")
tic = time.time()
predictor.set_image(img)
toc = time.time()
print(f"Done: {toc - tic:.2f} seconds")

# Try with a single point
pts = np.array([(312, 218)])  # from matplotlib hover feature
labels = np.ones(pts.shape[0])  # Assuming all points belong to cells
print("Running prediction...")
tic = time.time()
mask, C, logits = predictor.predict(pts, point_labels=labels)
toc = time.time()
print(f"Done: {(toc - tic)*1e3:.2f} ms")

# Visualize
fig, ax = plt.subplots(1, mask.shape[0])
for idx, (a, c, m) in enumerate(zip(ax, C, mask)):
    a.imshow(img)
    a.imshow(m, alpha=0.2)
    a.set_title(f"Mask {idx}, confidence: {c:.2f}")
plt.show()
