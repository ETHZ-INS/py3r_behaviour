from py3r.behaviour.config import DEV_MODE, USE_DISCONTINUED_METHODS
import functools


def dev_mode(func):
    """Decorator to mark a function as development-only."""
    # Handle staticmethod/classmethod
    is_static = isinstance(func, staticmethod)
    is_class = isinstance(func, classmethod)
    original_func = func.__func__ if (is_static or is_class) else func

    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        if not DEV_MODE:
            raise RuntimeError(
                f"{original_func.__name__} is not available in production mode."
            )
        return original_func(*args, **kwargs)

    wrapper.__doc__ = f"[DEV MODE ONLY]\n{original_func.__doc__ or ''}"
    if is_static:
        return staticmethod(wrapper)
    if is_class:
        return classmethod(wrapper)
    return wrapper


def discontinued_method(func):
    """Decorator to mark a method as discontinued."""
    is_static = isinstance(func, staticmethod)
    is_class = isinstance(func, classmethod)
    original_func = func.__func__ if (is_static or is_class) else func

    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        if not USE_DISCONTINUED_METHODS:
            raise RuntimeError(
                f"{original_func.__name__} is discontinued and no longer in use."
            )
        return original_func(*args, **kwargs)

    wrapper.__doc__ = f"[DISCONTINUED METHOD]\n{original_func.__doc__ or ''}"
    if is_static:
        return staticmethod(wrapper)
    if is_class:
        return classmethod(wrapper)
    return wrapper
