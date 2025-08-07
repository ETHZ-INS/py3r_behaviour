DEV_MODE = True  # Set to False in production


def dev_mode(func):
    """Decorator to mark a function as development-only."""

    def wrapper(*args, **kwargs):
        if not DEV_MODE:
            raise RuntimeError(f"{func.__name__} is not available in production mode.")
        return func(*args, **kwargs)

    wrapper.__doc__ = f"[DEV MODE ONLY]\n{func.__doc__ or ''}"
    return wrapper
