import copy
import importlib.util
import sys
import types
from pathlib import Path
import pytest
from percinterp.e08_io import atomic_json, read_json, accept_reviewed_repair


def test_tokenizer_uses_local_snapshot_and_never_queries_a_model_id(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'experiments'))
    import e08_collect
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    for name in ('tokenizer_config.json', 'tokenizer.json', 'config.json'):
        (snapshot / name).write_text('{}')
    hub = types.ModuleType('huggingface_hub')
    hub.try_to_load_from_cache = lambda *args: str(snapshot / 'tokenizer_config.json')
    transformer = types.ModuleType('transformers')
    class Tokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert Path(path) == snapshot and Path(path).is_dir()
            assert kwargs == {'local_files_only': True, 'fix_mistral_regex': False}
            return 'local tokenizer'
    transformer.AutoTokenizer = Tokenizer
    monkeypatch.setitem(sys.modules, 'huggingface_hub', hub)
    monkeypatch.setitem(sys.modules, 'transformers', transformer)
    assert e08_collect.load_cached_tokenizer() == 'local tokenizer'
    hub.try_to_load_from_cache = lambda *args: None
    with pytest.raises(FileNotFoundError, match='no download'):
        e08_collect.load_cached_tokenizer()


def test_reviewed_implementation_repair_requires_exact_unchanged_design_and_saves_history(tmp_path):
    old = {'design_sha256': 'fixed science', 'code_sha256': {'collect.py': 'old'}}
    new = {'design_sha256': 'fixed science', 'code_sha256': {'collect.py': 'repaired'}}
    path = tmp_path / 'protocol.json'
    receipt_path = tmp_path / 'receipt.json'
    atomic_json(path, old)
    receipt = {'previous_manifest': old, 'new_manifest': new, 'changed_files': ['collect.py'],
        'reason': 'offline tokenizer cache resolution', 'verification': 'CPU offline test passed',
        'reviewer': 'Codex implementation review', 'scientific_settings_unchanged': True}
    invalid = copy.deepcopy(receipt)
    invalid['new_manifest']['design_sha256'] = 'changed science'
    atomic_json(receipt_path, invalid)
    with pytest.raises(RuntimeError, match='unchanged-design'):
        accept_reviewed_repair(path, invalid['new_manifest'], receipt_path)
    assert read_json(path) == old
    invalid = copy.deepcopy(receipt); invalid['changed_files'] = []
    atomic_json(receipt_path, invalid)
    with pytest.raises(RuntimeError, match='changed-file'):
        accept_reviewed_repair(path, new, receipt_path)
    atomic_json(receipt_path, receipt)
    accept_reviewed_repair(path, new, receipt_path)
    assert read_json(path) == new
    history = list((tmp_path / 'protocol_history').glob('*.json'))
    assert len(history) == 1 and read_json(history[0]) == old
    assert read_json(tmp_path / 'last_reviewed_repair.json')['receipt_sha256']
