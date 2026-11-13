
from pathlib import Path
from tempfile import NamedTemporaryFile

from itertools import combinations
from math import comb
from typing import Iterator, Literal #'Self' only Python>3.11
from typing_extensions import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, PrivateAttr

# Each side lists the feature indices to multiply. Repeating an index squares it.
Relation = tuple[tuple[int, ...], tuple[int, ...]]



def _validate_samples(samples: ArrayLike) -> NDArray[np.float64]:
    try:
        X = np.asarray(samples)
    except (TypeError, ValueError) as exc:
        raise ValueError("samples must be a rectangular 2D array.") from exc

    if X.ndim != 2:
        raise ValueError(f"samples must be 2D, got shape {X.shape}.")
    if X.shape[0] == 0 or X.shape[1] < 2:
        raise ValueError("samples must have at least one row and two features.")
    if X.dtype.kind not in "iuf":
        raise ValueError("samples must contain real numbers.")

    with np.errstate(over="ignore", under="ignore"):
        X = X.astype(np.float64, copy=False)

    if not np.isfinite(X).all():
        raise ValueError("Every entry in samples must be finite in float64.")
    #if np.any(X <= 0):
    #    raise ValueError("Every entry in samples must be strictly positive.")

    return X



def _relations(n: int) -> Iterator[Relation]:
    """Yield comparisons in the original training order."""
    # The yield order defines the position of each relation in the saved file.
    for i, j in combinations(range(n), 2):
        yield (i,), (j,)  # x_i < x_j

        # Compare each pair product with the square of a third feature.
        for ell in range(n):
            if ell != i and ell != j:
                yield (i, j), (ell, ell)  # x_i * x_j < x_ell**2

        # Use disjoint pairs and put the smallest index on the left to avoid reversals.
        remaining = (k for k in range(i + 1, n) if k != j)
        for h, k in combinations(remaining, 2):
            yield (i, j), (h, k)  # x_i * x_j < x_h * x_k



def _compare(
    X: NDArray[np.float64],
    relation: Relation,
    rows: slice | NDArray[np.intp] = slice(None),
) -> NDArray[np.bool_]:
    # Evaluate the strict comparison for every requested row in one NumPy operation.
    left, right = relation
    if len(left) == 1:
        return X[rows, left[0]] < X[rows, right[0]]

    # For a product comparison, the same formula also handles a squared feature.
    i, j = left
    h, k = right
    return X[rows, i] * X[rows, j] < X[rows, h] * X[rows, k]




def _count_relations(n: int) -> int:
    # Each pair gives one direct comparison. Each triple gives three square comparisons.
    # Each quadruple has three ways to split its indices into two disjoint pairs.
    return comb(n, 2) + 3 * comb(n, 3) + 3 * comb(n, 4)



class MOND(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Store the feature count and one outcome code per relation in generation order.
    _trained: tuple[int, NDArray[np.int8]] | None = PrivateAttr(default=None)



    def train(
        self, samples: ArrayLike, *, relations_path: str | Path | None = None
    ) -> Self:
        """Learn fixed inequality outcomes and optionally save them as text."""
        X = _validate_samples(samples)
        n = X.shape[1]
        r = _count_relations(n) #comb(n, 2) + 3 * comb(n, 3) + 3 * comb(n, 4)
        # Keep only the outcome code for each relation, without storing a sample-by-relation matrix.
        c = np.empty(r, dtype=np.int8)

        # Reduce one Boolean column at a time to its outcome across all training rows.
        with np.errstate(over="raise", invalid="raise"):
            for s, relation in enumerate(_relations(n)):
                column = _compare(X, relation)
                if column.all():
                    c[s] = 1   # Always true
                elif column.any():
                    c[s] = -1  # Varies across training rows
                else:
                    c[s] = 0   # Always false

        if relations_path is not None:
            path = Path(relations_path)
            # Write beside the destination so the completed file can replace it atomically.
            with NamedTemporaryFile(
                mode="w", encoding="ascii", dir=path.parent, delete=False
            ) as output:
                temporary_path = Path(output.name)
                try:
                    # Save one ASCII integer per line in the order expected by load_relations.
                    np.savetxt(output, c, fmt="%d")
                    output.close()
                    temporary_path.replace(path)
                finally:
                    temporary_path.unlink(missing_ok=True)

        # Publish the fitted state after the optional file write succeeds.
        self._trained = (n, c)
        return self



    def load_relations(self, path: str | Path) -> Self:
        """Load relation codes and infer the feature count from their number."""
        lines = Path(path).read_bytes().splitlines()
        codes = {b"-1": -1, b"0": 0, b"1": 1}
        c = np.empty(len(lines), dtype=np.int8)

        for index, line in enumerate(lines):
            code = line.strip()
            if code not in codes:
                raise ValueError(f"Line {index + 1}: expected -1, 0, or 1.")
            c[index] = codes[code]

        # Each feature count has a unique number of relations.
        n_features = 2
        while _count_relations(n_features) < c.size:
            n_features += 1
        if _count_relations(n_features) != c.size:
            raise ValueError(
                f"{c.size} relation codes do not match any feature count."
            )

        # Restore the same fitted state as train() so prediction can start immediately.
        self._trained = (n_features, c)
        return self



    def predict(
        self,
        samples: ArrayLike,
        *,
        mode: Literal["eager", "lazy"] = "eager",
        max_non_verified: int | None = None,
    ) -> NDArray[np.bool_]:
        """Classify rows by how many fixed relations they fail.

        Eager mode rejects a row at its first failure. Lazy mode evaluates
        every fixed relation and accepts rows with at most max_non_verified
        failures. The threshold is required only in lazy mode.
        """
        if mode not in ("eager", "lazy"):
            raise ValueError("mode must be 'eager' or 'lazy'.")

        if mode == "lazy":
            if max_non_verified is None:
                raise ValueError("max_non_verified is required in lazy mode.")
            if isinstance(max_non_verified, bool) or not isinstance(
                max_non_verified, (int, np.integer)
            ):
                raise TypeError("max_non_verified must be a positive integer.")
            if max_non_verified <= 0:
                raise ValueError("max_non_verified must be a positive integer.")
        trained = self._trained
        if trained is None:
            raise RuntimeError("Call train() before predict().")

        n, c = trained
        raw = np.asarray(samples)
        if raw.ndim != 2 or raw.shape[0] == 0:
            raise ValueError("samples must be a nonempty 2D array.")
        if raw.shape[1] != n:
            raise ValueError(f"Expected {n} features, got {raw.shape[1]}.")

        # Keep the trained True (1) and False (0) outcomes. Skip variable (-1) outcomes.
        fixed = [
            (relation, bool(expected))
            for expected, relation in zip(c, _relations(n), strict=True)
            if expected >= 0
        ]
        # A row can fail at most as many relations as there are fixed outcomes.
        failure_limit = (
            min(int(max_non_verified), len(fixed))
            if mode == "lazy" and max_non_verified is not None
            else None
        )

        # Rejected rows keep their initial False value.
        result = np.zeros(raw.shape[0], dtype=bool)
        # Bound the amount of data validated and processed at once.
        block_size = 65_536

        with np.errstate(over="raise", invalid="raise"):
            for start in range(0, raw.shape[0], block_size):
                stop = min(start + block_size, raw.shape[0])
                # Store each feature contiguously for column-wise comparisons.
                X = np.asfortranarray(_validate_samples(raw[start:stop]))

                if failure_limit is not None:
                    # Keep one failure count per row in this block. Every fixed
                    # relation is checked even if a row has exceeded the limit.
                    failures = np.zeros(stop - start, dtype=np.intp)
                    for relation, expected in fixed:
                        failures += _compare(X, relation) != expected
                    result[start:stop] = failures <= failure_limit
                    continue

                # All rows in this block start as candidates. A full slice avoids an index array.
                rows = slice(None)
                for relation, expected in fixed:
                    # Check this relation only for rows that passed every earlier check.
                    keep = _compare(X, relation, rows) == expected
                    if keep.all():
                        continue

                    # Keep block-relative indices so surviving rows retain their original positions.
                    if isinstance(rows, slice):
                        rows = np.flatnonzero(keep)
                    else:
                        rows = rows[keep]

                    # Stop checking this block once every row has failed a relation.
                    if rows.size == 0:
                        break

                # This slice views the output array. Mark only the surviving rows True.
                block_result = result[start:stop]
                block_result[rows] = True

        return result