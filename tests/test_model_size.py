import ast
from pathlib import Path

from finetune.utils.training_utils import get_model_size


ROOT = Path(__file__).resolve().parents[1]


class _Params:
    def __init__(self, total_params: int):
        self._total_params = total_params
        self.requires_grad = True

    def numel(self) -> int:
        return self._total_params


class _Model:
    def __init__(self, total_params: int):
        self._total_params = total_params

    def parameters(self):
        return [_Params(self._total_params)]


def _load_csv_get_model_size():
    source = ROOT / "finetune_csv" / "finetune_tokenizer.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_model_size"
    ]
    # The annotation names torch only; the function body does not use it.
    torch_stub = type("torch", (), {"nn": type("nn", (), {"Module": object})})()
    namespace = {"torch": torch_stub}
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["get_model_size"]


def test_get_model_size_steps_up_when_rounding_reaches_1000():
    assert get_model_size(_Model(999_950)) == "1.0M"
    assert get_model_size(_Model(999_949)) == "999.9K"
    assert get_model_size(_Model(999_950_000)) == "1.0B"
    assert get_model_size(_Model(1_000_000)) == "1.0M"
    assert get_model_size(_Model(1_000_000_000)) == "1.0B"


def test_get_model_size_does_not_round_tiny_counts_to_zero_k():
    assert get_model_size(_Model(0)) == "0"
    assert get_model_size(_Model(49)) == "49"
    assert get_model_size(_Model(50)) == "0.1K"
    assert get_model_size(_Model(7_100_000)) == "7.1M"


def test_csv_copy_matches_shared_formatter():
    get_model_size_csv = _load_csv_get_model_size()
    for n in (49, 999_950, 999_950_000):
        model = _Model(n)
        assert get_model_size_csv(model) == get_model_size(model)
