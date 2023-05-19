# Trying out segment_anything

To setup:

```bash
python -m venv sam-env
source sam-env/bin/activate
pip install -r requirements.txt
```

### Hello world

`samlog.py`: Example of running model programmatically with a single-cell image
(nuclear + membrane channels).

To run interactively:

1. Run `ipython -i samlog.py`
2. There is a breakpoint to allow you to interact with the image.
   Left-click to add "object" points and right-click to add "not object" points
3. When you're done adding points, type `c` + `enter` to exit the debugger.

You can continue interacting with the image and re-running predictions as well.

Derived from [the napari plugin from JoOkuma](https://github.com/JoOkuma/napari-segment-anything)
