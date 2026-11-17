

# MOND

Mutual Ordering of Normalized Differences (MOND) is a NumPy-based one-class classifier that learns which feature-based inequalities hold consistently across examples of a target class, see [1]. It trains on target-class samples alone and returns either a boolean prediction or a count of failed relations per input row. For raster classification, a row can represent a pixel and its columns can represent bands or other features.

The MOND classifier is described and applied to remote sensing multispectral images in this [Research Paper](https://massimozanetti.github.io/assets/pdf/publications/zanetti2023oneclass.pdf). 

Run the MOND classifier on Sentinel-2 images in this [GEE App](https://massimozanettifbk.users.earthengine.app/view/s2-oneclass-mond).



## How the classifier works

### Training
The input for training is a sample set $`x^{(m)}=(x^{(m)}_1, \ldots, x^{(m)}_N)`$, for $m=1, \ldots, M$, with $M$ the number of samples (the rows of the input array) and $N$ the number of features (the columns of the input array), representing the target class.

MOND evaluates on each sample every strict inequality of the type $`x_i x_j < x_k x_h`$ with subscripts $`i, j, k, h \in \{1, ..., N\}`$ fulfilling the constraints $i \leq j$ and $i < k \leq h$, with both $k, h \neq j$.   

Training evaluates each inequality relation across all training samples and stores one outcome code. A code of `1` means the inequality was always true, `0` means it was always false, and `-1` means its outcome varied across samples. 

### Prediction

Prediction checks whether each input row agrees with the fixed outcomes learned during training. Fixed relations have code `1` (always true) or `0` (always false). Prediction skips relations with code `-1`.

| Argument | Default | Description |
| --- | --- | --- |
| `eager` | `True` | With `True`, return one Boolean per row and stop checking a row at its first disagreement. With `False`, check every fixed relation and return one integer failure count per row. |
| `block_size` | `65_536` | Maximum number of input rows processed in a block in either mode. Provide a positive integer. |
| `n_cpu` | `None` | Worker limit for lazy mode. With `None`, use the detected CPU count. Pass `1` for serial execution or a larger positive integer to process relation groups in parallel. The worker count is capped by the number of fixed relations. |

A lazy count of `0` means every fixed relation agrees with its training outcome. If training finds no fixed relations, each valid row returns `True` in eager mode and `0` in lazy mode.



## Installation

Use Python 3.10 or newer and install the dependencies:

```bash
python -m pip install numpy "pydantic>=2,<3" "typing-extensions>=4.0"
```

Run the examples from the directory containing [MOND.py](MOND.py).



## Quick start

Basic usage with in-memory trained relations: 

```python
import numpy as np

from MOND import MOND

train_samples = np.array([[1, 2, 3], [2, 4, 6]])
test_samples = np.array([[1, 2, 3], [1, 2, 2], [3, 2, 1]])

model = MOND().train(train_samples)

classif = model.predict(test_samples)

print(classif.tolist())  # [True, False, False]
```

Each input must be a nonempty two-dimensional array of size `(n_samples, n_features)` made of finite non-negative real numbers.



## Save and reuse trained relations

Pass `relations_path` to `train()` to save trained relations: 

```python
model = MOND().train(train_samples, relations_path="relations.dat")
```

To reuse the saved relations:

```python
model = MOND().load_relations("relations.dat")

classif = model.predict(test_samples)
print(classif.tolist())  # [True, False, False]
```

The loader infers the feature count from the number of codes and raises `ValueError` if the count matches no supported number of features. The file stores relation outcomes in a fixed order, so use the original feature order when predicting with a loaded model.



## Eager and lazy modes

Use the same test samples to get Boolean predictions or count how many fixed relations each sample fails.

```python
verified = model.predict(test_samples, eager=True)
counts = model.predict(test_samples, eager=False, block_size=32_768, n_cpu=4)

print(verified.tolist())  # [True, False, False]
print(counts.tolist())    # [0, 1, 5]
```

Pass `block_size` to adjust how many pixels are processed in each block. To produce an array of failure counts, pass `eager=False` to `model.predict()`.



## Predict on a raster

For an array with shape `(height, width, n_features)`, flatten the spatial dimensions before prediction and reshape the Boolean output back to the raster grid:

```python
def predict_raster(model, image):
    height, width, n_features = image.shape
    samples = image.reshape(height * width, n_features)
    return model.predict(samples).reshape(height, width)
```



## Reference

[1] Zanetti, M., (2023). “A One-class classification model for burned-area detection based on mutual order of normalized differences.” *IEEE Transactions on Geoscience and Remote Sensing*.