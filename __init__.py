success = False
try:
    from src import *
except ModuleNotFoundError as e:
    try:
        from .src import *
        success = True
    except Exception as f:
        raise f
    if not success:
        raise e
