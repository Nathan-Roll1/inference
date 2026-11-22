"""Run beside the stock pip package. No GPU, model weights or simulator are needed."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from unittest.mock import patch

import requests

url = 'http://127.0.0.1:9997/v1'
model = json.loads(Path('fourier-vllm-registration.json').read_text())
name = model['model_name']
receipt = {'xinference': importlib.metadata.version('xinference'), 'sessions': []}


def save():
    Path('receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


def catalog_identity():
    import xinference
    root = Path(xinference.__file__).parent / 'model/audio/models'
    assert not (root / (name + '.json')).exists()
    return hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in sorted(root.glob('*.json')))).hexdigest()


def session(index):
    with open(f'server-{index}.log', 'w') as log:
        proc = subprocess.Popen(['xinference-local', '--host', '127.0.0.1', '--port', '9997'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            for i in range(180):
                if proc.poll() is not None:
                    raise RuntimeError(Path(log.name).read_text()[-4000:])
                try:
                    if requests.get(url+'/models',timeout=3).ok:
                        break
                except requests.ConnectionError:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError(Path(log.name).read_text()[-4000:])
            record = {'server_pid': proc.pid}
            if index == 1:
                r = requests.post(url+'/model_registrations/audio',json={'model': json.dumps(model), 'persist':True},timeout=45)
                record['post_status'] = r.status_code
                print('POST registration', r.status_code, r.text[:800], flush=True)
                r.raise_for_status()
            r = requests.get(url+'/model_registrations/audio/'+name,timeout=30)
            print('GET registration', r.status_code, r.text[:1100], flush=True)
            r.raise_for_status()
            obj=r.json()
            for key in ('model_id','model_revision','engine','model_family','model_format'):
                assert obj[key] == model[key], (key,obj)
            assert obj.get('model_uri') is None
            assert not obj.get('is_builtin')
            record['registration']={key:obj.get(key) for key in ('model_name','model_id','model_revision','model_uri','engine','model_family','model_format','is_builtin')}
            r = requests.get(url+'/engines/audio/'+name,timeout=30)
            record['cpu_engine_discovery'] = {'status':r.status_code,'body':r.text[:3000]}
            receipt['sessions'].append(record)
            save()
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
            try:
                proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
            try:
                os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass


before = catalog_identity()
session(1)
session(2)
assert before == catalog_identity()
receipt['builtin_catalog_sha256'] = before
folder = Path(os.environ['XINFERENCE_HOME'])/'model/v2/audio'
specfiles=list(folder.glob('*'))
assert len(specfiles)==1
saved = json.loads(specfiles[0].read_text())
assert saved['model_id']==model['model_id'] and saved['model_revision']==model['model_revision']
receipt['persisted_config_filename'] = specfiles[0].name

from xinference.model import audio
from xinference.model.audio import engine
from xinference.model.audio.core import create_audio_model_instance,match_audio
from xinference.model.audio.engine_family import AUDIO_ENGINES
from xinference.model.audio.custom import get_user_defined_audios
from xinference.model.audio.qwen3_asr import Qwen3ASRModel

# Runtime constructors accept a supplied model_path. They do not read it until load().
# Only checking selection here; no loading, inference, or GPU claim.
audio.register_builtin_model()
assert any(f.model_name==name for f in get_user_defined_audios())
spec = match_audio(name,model_engine='vLLM')
assert engine.has_cuda_device() is False
args=dict(model_uid='oruk-fourier-2b',model_name=name,model_engine='vLLM',enable_virtual_env=False,
          max_model_len=1024,max_num_seqs=2,gpu_memory_utilization=0.7,enforce_eager=True,max_new_tokens=96,
          model_path='/nonexistent-no-weights-used')
cpu_instance = create_audio_model_instance(**args)
receipt['cpu_control_class'] = type(cpu_instance).__name__
assert isinstance(cpu_instance,Qwen3ASRModel)
assert not isinstance(cpu_instance,engine.VLLMQwen3ASRAudioModel)
assert engine.VLLMQwen3ASRAudioModel.is_model_family_supported(spec)
# A single mocked hardware predicate exercises unmodified stock registry/dispatch code
# with the persisted custom spec on the CPU runner, explicitly separate from the CPU control.
with patch.object(engine,'has_cuda_device',return_value=True):
    from xinference.model.audio.engine_family import generate_engine_config_by_model_name
    generate_engine_config_by_model_name(spec)
    receipt['conditional_engine_names'] = list(AUDIO_ENGINES[name])
    selected = create_audio_model_instance(**args)
    receipt['conditional_selector_class'] = type(selected).__name__
    receipt['conditional_only_mock'] = 'xinference.model.audio.engine.has_cuda_device=True'
    assert isinstance(selected,engine.VLLMQwen3ASRAudioModel)
    assert not isinstance(selected,Qwen3ASRModel)
receipt['model_loaded'] = False
receipt['local_weights_present'] = False
save()
print(json.dumps(receipt,indent=2))
