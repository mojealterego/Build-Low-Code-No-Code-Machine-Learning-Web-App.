"""Legacy compatibility entry point.

The maintained application lives in ``app.py``. Keeping this small wrapper
prevents two divergent Streamlit implementations from drifting apart.
"""

from app import main


if __name__ == "__main__":
    main()
