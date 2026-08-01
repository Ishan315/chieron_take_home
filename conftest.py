import sys
from pathlib import Path

# Add backend folder to Python module path
root_dir = Path(__file__).parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
