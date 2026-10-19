import pytest
import torch
from hivemind.swarm.dht_storage import ContentAddressedSwarmStore


def test_storage_proof_binds_file_bytes(tmp_path):
    store = ContentAddressedSwarmStore(tmp_path)
    cid, path, _ = store.store_hf_dataset_shard('n', 'test', 0, ['original'])
    assert store.verify_proof_of_storage('n', cid, path, 'seed').verified
    path.write_bytes(b'tampered')
    assert not store.verify_proof_of_storage('n', cid, path, 'seed').verified
    with pytest.raises(ValueError):
        store.verify_proof_of_storage('n', cid, path, 'seed', chunk_size=0)

@pytest.mark.parametrize('kind', ['tensor', 'name', 'shape', 'metadata', 'cid'])
def test_checkpoint_integrity_binds_tensor_semantics(tmp_path, kind):
    store = ContentAddressedSwarmStore(tmp_path)
    cid, path = store.publish_lora_expert_checkpoint('test', {'a':torch.ones(2,2)}, {'step':1})
    assert store.load_lora_expert_checkpoint(cid)[0] == cid
    payload = torch.load(path, weights_only=True)
    if kind == 'tensor': payload['lora_state']['a'][0,0] = 2
    if kind == 'name': payload['lora_state']['b'] = payload['lora_state'].pop('a')
    if kind == 'shape': payload['lora_state']['a'] = payload['lora_state']['a'].reshape(4)
    if kind == 'metadata': payload['metadata']['step'] = 2
    if kind == 'cid': payload['cid'] = 'cid://sha256/not-original'
    torch.save(payload, path)
    with pytest.raises(ValueError): store.load_lora_expert_checkpoint(cid)

def test_domain_cannot_escape_storage_directory(tmp_path):
    store = ContentAddressedSwarmStore(tmp_path)
    with pytest.raises(ValueError): store.store_hf_dataset_shard('n', '../bad', 0, ['x'])
    with pytest.raises(ValueError): store.publish_lora_expert_checkpoint('../bad', {'x':torch.ones(1)}, {})
