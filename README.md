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

A programmatic approach based on
[the napari plugin from JoOkuma](https://github.com/JoOkuma/napari-segment-anything)
