from pathlib import Path

DEFAULT_DIST_PATH = Path('/srv/www/cozy/dist')


def get_web_root() -> Path:
    """
    Utility function to find the web root directory for the static file server
    """
    dev_path = Path(__file__).parent.parent.parent.parent.parent.parent / 'web' / 'dist'
    prod_path = DEFAULT_DIST_PATH
    
    if dev_path.exists():
        return dev_path
    elif prod_path.exists():
        return prod_path
    else:
        raise FileNotFoundError("Neither primary nor secondary web root paths exist.")