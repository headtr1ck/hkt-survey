# `xarray.NamedArray` / `xarray.DataArray`: wrappers around arbitrary arrays

Sources:

- [xarray/namedarray/core.py][src-namedarray], [xarray/namedarray/_array_api.py][src-astype]
- pydata/xarray#11712 (draft): [generic `DataArray`][src-dataarray], [`astype`][src-dataarray-astype], [typing test for subclasses][src-test]

xarray's `NamedArray`, `Variable` and `DataArray` wrap an arbitrary duck array (NumPy, CuPy, JAX, sparse, pint, ...) and add named dimensions. Methods transform the wrapped array and rebuild the result with `type(self)`, so at runtime both the *subclass* and the *wrapped array type* are preserved, while operations like `astype` or `==` change the shape or dtype. The same pattern applies to any library that wraps arbitrary arrays.

This file shows two designs: the one xarray uses today, which needs HKT only for subclasses, and the one it would use if HKT existed, which needs two nested higher-kinded variables.

## Today's design: generic in shape and dtype

`DataArray[ShapeT, DTypeT, DimT, VarT]` is generic in the shape and dtype directly, and types the wrapped array as a `duckarray[ShapeT, DTypeT]` protocol. Methods that keep all type arguments return `Self`. Methods that change one can't keep the subclass, which is the pattern from [scipy_sparse_astype.md](scipy_sparse_astype.md):

Kind: `Origin[Self]` is $\ast \to \ast \to \ast \to \ast \to \ast$, re-applied with the dtype swapped.

```python
from typing import Any, Origin, Self

class DataArray[ShapeT, DTypeT, DimT, VarT]:
    def __neg__(self) -> Self: ...  # works today

    def astype[D, *Ts: _](
        self: Origin[Self][ShapeT, Any, *Ts],
        dtype: D,
    ) -> Origin[Self][ShapeT, D, *Ts]: ...

class MyDataArray[ShapeT, DTypeT, DimT, VarT](DataArray[ShapeT, DTypeT, DimT, VarT]): ...
```

```python
my: MyDataArray[tuple[int], np.dtype[np.float64], str, str]
reveal_type(-my)  # MyDataArray[tuple[int], np.dtype[np.float64], str, str]
reveal_type(my.astype(np.dtype(np.int32)))  # MyDataArray[tuple[int], np.dtype[np.int32], str, str]
# today: DataArray[tuple[int], np.dtype[np.int32], str, str]
```

The typing tests in xarray document this: "they keep subclasses, which type checkers cannot express together with the changed types (it would need higher-kinded types)".

## The design HKT would allow: generic in the array type

The protocol for the wrapped array is itself a workaround: `.data` is typed as `duckarray[ShapeT, DTypeT]` and loses the concrete array type, although the docstring promises "The underlying array type (e.g. dask, sparse, pint) is preserved". Ideally the wrapper would be generic in the wrapped array type, `NamedArray[ArrayT, DimT]`, with shape and dtype derived from `ArrayT`. Unlike the other examples here, the class is an ordinary generic over a *fully applied* type, and the higher-kinded variables are recovered from `self` and from a type argument of `self`.

Kind: `Origin[Self]` is $\ast \to \ast \to \ast$ and `A` is $\ast \to \ast \to \ast$, nested as `Origin[Self][A[S, D], *Ts]`.

```python
from typing import Any, Origin, Protocol, Self
import numpy as np

class duckarray[ShapeT_co, DTypeT_co](Protocol):
    @property
    def shape(self) -> ShapeT_co: ...
    @property
    def dtype(self) -> DTypeT_co: ...

class NamedArray[ArrayT: duckarray[Any, Any], DimT]:
    _data: ArrayT

    def __init__(self, dims: tuple[DimT, ...], data: ArrayT) -> None: ...

    # works today: read the type arguments through the protocol
    @property
    def data(self) -> ArrayT: ...
    @property
    def dtype[D](self: NamedArray[duckarray[Any, D], Any]) -> D: ...

    # works today: nothing changes
    def __neg__(self) -> Self: ...

    # needs HKT twice: the subclass and the wrapped array's constructor
    def astype[A: Origin[duckarray], S, D, *Ts: _](
        self: Origin[Self][A[S, Any], *Ts],
        dtype: D,
    ) -> Origin[Self][A[S, D], *Ts]:
        xp = self._data.__array_namespace__()
        return self._new(data=xp.astype(self._data, dtype))

class Variable[ArrayT: duckarray[Any, Any], DimT](NamedArray[ArrayT, DimT]): ...
```

`A` needs the `Origin[duckarray]` bound so that `A[S, D]` (with the new `D`) still satisfies the bound on `ArrayT`. For the body to type-check, `xp.astype` must itself return "`A` re-applied with `D`", the same problem one level down.

```python
x = Variable(("x",), np.zeros(3))
reveal_type(x)  # Variable[np.ndarray[tuple[int], np.dtype[np.float64]], str]
reveal_type(x.dtype)  # np.dtype[np.float64]
reveal_type(-x)  # Variable[np.ndarray[tuple[int], np.dtype[np.float64]], str]
reveal_type(x.astype(np.dtype(np.int32)))  # Variable[np.ndarray[tuple[int], np.dtype[np.int32]], str]

import dask.array as da

y = Variable(("x",), da.zeros(3))
reveal_type(y.astype(np.dtype(np.int32)))  # Variable[da.Array, str]
```

Wrapped array types disagree on arity: `np.ndarray` is binary (shape, dtype), while e.g. `dask.array.Array` or `sparse.COO` aren't generic at all ($\ast$). For those, `astype` can't re-apply anything and should keep the array type, as in the dask call above; today that fallback can be written as an overload. Array libraries may also order their type parameters differently from `duckarray[ShapeT, DTypeT]`, in which case re-applying `A[S, D]` by position is wrong (see #8).

## Without HKT

- Overloads per known array type (`self: NamedArray[np.ndarray[S, Any], DimT]`) work, but only for a closed set of array types, and they return the base class, not the subclass.
- Borrowing the return type from the wrapped array's own method via a protocol on `self` (`self: NamedArray[SupportsAstype[D, R], DimT]` → `NamedArray[R, DimT]`) works in pyright even for NumPy's overloaded `ndarray.astype`, but mypy infers `Never` there. It also only helps when the wrapped array has an equivalent, precisely typed method: the array API defines `astype` as a namespace function, and many xarray operations are composed from several array functions rather than delegated to one method.

[src-namedarray]: https://github.com/pydata/xarray/blob/357efad180e54c3e9c419c5d441e68f06e2006eb/xarray/namedarray/core.py#L313-L351
[src-astype]: https://github.com/pydata/xarray/blob/357efad180e54c3e9c419c5d441e68f06e2006eb/xarray/namedarray/_array_api.py#L31-L72
[src-dataarray]: https://github.com/pydata/xarray/blob/bb6e61c9fa892b2bcf93475bf6d8a9b7f4696b42/xarray/core/dataarray.py#L320-L326
[src-dataarray-astype]: https://github.com/pydata/xarray/blob/bb6e61c9fa892b2bcf93475bf6d8a9b7f4696b42/xarray/core/dataarray.py#L522-L531
[src-test]: https://github.com/pydata/xarray/blob/bb6e61c9fa892b2bcf93475bf6d8a9b7f4696b42/xarray/tests/test_typed_ops_typing.yml#L121-L139
