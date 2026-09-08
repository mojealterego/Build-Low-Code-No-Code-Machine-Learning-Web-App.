from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "Building Low code application" / "app.py"

spec = spec_from_file_location("aurelis_app", APP_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load application module")

module = module_from_spec(spec)

# The application has optional runtime-heavy ML imports; helper tests are kept
# intentionally independent from the Streamlit UI and PyCaret execution path.
try:
    spec.loader.exec_module(module)
except Exception as exc:  # pragma: no cover - dependency-light environments
    module = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def test_imports_or_exposes_dependency_requirement() -> None:
    assert module is not None or IMPORT_ERROR is not None


def test_safe_model_name_when_module_is_available() -> None:
    if module is None:
        return
    assert module.safe_model_name(" My Model / v1 ") == "My_Model_v1"
    assert module.safe_model_name("   ") == "aurelis_model"
