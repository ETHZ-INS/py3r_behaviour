from __future__ import annotations

from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Iterator


@contextmanager
def data_path(package: str, resource: str) -> Iterator[Path]:
    """
    Yield a filesystem path to a packaged data file.

    This helper works for both editable installs and wheels and is intended
    for use in doctests and examples where a real file path is needed.

    Parameters
    ----------
    package : str
        Dotted package path that contains the resource directory.
    resource : str
        Relative filename within the package.

    Examples
    --------
    >>> with data_path('py3r.behaviour.tracking._data', 'yolo3r.csv') as p:
    ...     p.name.endswith('.csv')
    True
    """
    with resources.as_file(resources.files(package).joinpath(resource)) as p:
        yield Path(p)
