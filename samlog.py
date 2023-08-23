from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, color, util

from segment_anything import SamPredictor, sam_model_registry
from cursor import InteractivePredictor

plt.ion()

def tissuenet_to_sam_input(img):
    img = img.sum(axis=-1)  # Combine nuclear and membrane into single channel
    img = color.gray2rgb(img)  # Convert to rgb TODO: is this necessary for model?
    img -= img.min()  # fp from 0. to 1.
    img /= img.max()
    img = util.img_as_ubyte(img)
    return img.transpose((1, 0, 2))

# Setup SAM
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

# Load tissuenet
datapath = Path.home() / ".deepcell/tissuenet_v1-1/test.npz"
data = np.load(datapath)
X = data["X"]
y = data["y"]

# Pick an image
idx = 200
img, mask = tissuenet_to_sam_input(X[idx]), y[idx][..., 0]

# Load image -> model
print("Loading image to SAM...")
predictor.set_image(img)
print("Done.")

# Plot image
fig, ax = plt.subplots()
ax.imshow(img)

# Get centroids (for demonstration purposes)
labels = measure.regionprops(mask)
centroids = np.array([p.centroid for p in labels])
ax.scatter(*np.array(centroids).T)

# Create an interactive SAM mpl app
app = InteractivePredictor(ax, predictor)
