# MOND

Mutual Ordering of Normalized Differences (MOND) is a NumPy-based one-class classifier that learns which feature-based inequalities hold consistently across examples of a target class. It trains on target-class samples alone and returns one Boolean prediction per input row. For raster classification, a row can represent a pixel and its columns can represent bands or other features.

The MOND classifier is described and applied to remote sensing multispectral images in this [Research Paper](https://massimozanetti.github.io/assets/pdf/publications/zanetti2023oneclass.pdf).

## Installation

Use Python 3.10 or newer and install the dependencies:

```bash
python -m pip install numpy "pydantic>=2,<3" "typing-extensions>=4.0"
```

Run the examples from the directory containing [MOND.py](MOND.py).

## Quick start

```python
import numpy as np

from MOND import MOND

training = np.array([[1, 2, 3], [2, 4, 6]])
samples = np.array([[1, 2, 3], [1, 2, 2], [3, 2, 1]])

model = MOND().train(training, relations_path="relations.dat")

eager = model.predict(samples)
lazy = model.predict(samples, mode="lazy", max_non_verified=1)

print(eager.tolist())  # [True, False, False]
print(lazy.tolist())   # [True, True, False]
```

Each input must be a nonempty two-dimensional array of size `(n_samples, n_features)` made of finite non-negative real numbers. Use the same number and order of features in training and prediction.

## How the classifier works

Let $N$ be the number of features. MOND evaluates every strict comparison $`x_i x_j < x_h x_k`$ for $`i, j, k, h \in \{1, ..., N\}`$ fulfilling the constraints $i \leq j$ and $i < k \leq h$, with both $k, h \neq j$.   

Training evaluates each relation across all training rows and stores one outcome code. A code of `1` means the inequality was always true, `0` means it was always false, and `-1` means its outcome varied. Prediction skips relations coded `-1`. Equality makes a strict comparison false, so it satisfies a relation coded `0`.

| Prediction mode | When a row returns `True` |
| --- | --- |
| `eager` (default) | Every fixed relation agrees with its stored outcome. Evaluation stops at the first disagreement. |
| `lazy` | The number of disagreements is at most `max_non_verified`, inclusive. Every fixed relation is evaluated. |

Lazy mode requires `max_non_verified` to be a positive integer. If training finds no fixed relations, every valid input row returns `True` in both modes.

## Save and reuse trained relations

Pass `relations_path` to `train()` to save trained relations (one code per line in relation order). Omit the argument for in-memory training. To reuse the saved relations:

```python
model = MOND().load_relations("relations.dat")
print(model.predict(samples).tolist())  # [True, False, False]
```

The loader infers the feature count from the number of codes and raises `ValueError` if the count matches no supported number of features. The file stores relation outcomes in a fixed order, so use the original feature order when predicting with a loaded model.

## Predict on a raster

For an array with shape `(height, width, n_features)`, flatten the spatial dimensions before prediction and reshape the Boolean output back to the raster grid:

```python
def predict_raster(model, image):
    height, width, n_features = image.shape
    samples = image.reshape(height * width, n_features)
    return model.predict(samples).reshape(height, width)
```

Prediction works in blocks of up to 65,536 rows. Eager mode can save computation when many pixels fail early. Lazy mode keeps one failure count per row in the current block and checks all fixed relations before applying the threshold.
