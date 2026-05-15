import hashlib
import json
from pathlib import Path


def test_runtime_model_matches_exported_threshold_artifact():
    artifact_path = Path('artifacts/thresholds.json')
    model_path = Path('model.pkl')

    assert artifact_path.exists(), 'Missing artifacts/thresholds.json'
    assert model_path.exists(), 'Missing model.pkl'

    artifact = json.loads(artifact_path.read_text())
    exported_hash = artifact.get('model_sha256')
    assert exported_hash, 'Artifact missing model_sha256'

    runtime_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert runtime_hash == exported_hash, (
        'Runtime model hash differs from exported threshold artifact hash for the same model version. '
        'Re-export model and thresholds together.'
    )
