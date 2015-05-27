# pybing

A Python wrapper of BING-Objectness in OpenCV (3.0.0-dev) implementation

## Requirement

- CMake (>=2.8)
- TBB
- OpenCV (>=3.0.0-dev with opencv_contrib)
- Boost
- Boost.Numpy

## Build bing.so

Modify some parameters to detect TBB, OpenCV, Boost, and Boost.NumPy, and then

  $ bash build.sh

It will make `bing.so` in `build` directory.

## Usage

- BING class requires the path of trained model dir and some parameters
- The output of `objectness` method is 2D-array of bounding boxes
  - `[[x1, y1, x2, y2, score],...]`
  - Smaller `score` means it has much objectness
  - Resulting bounding boxes is already sorted in ascending order of `score`, so it's descending order of objectness
- try `$ python scripts/test_bing.py`

```
import bing

b = 2  # base_window_size_quantization
w = 8  # window_size
n = 2  # non_maximal_supress_size

binger = bing.BING('build/ObjectnessTrainedModel', b, w, n)
img = cv.imread('sample.jpg')
canvas = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

bbox = binger.objectness(img)
for b in bbox:
    x1, y1, x2, y2 = [int(a) for a in b[:4]]
    s = b[-1]
    canvas[y1:y2, x1:x2] += s

canvas /= np.max(canvas)
cv.imwrite('sample_result.jpg', canvas * 255)
```
