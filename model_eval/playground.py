"""Configured agent jobs, held-out checks, immutable comparison records.

No physical allocator, queue transport, rail fork or automatic replacement job.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid
from .core import SCHEMA,digest,write_json
from .tasks import REVISION,TASKS,verify

REQUIRED=('model','compute','client','identity','service_id','resource_id','command','environment','profile_template','token_file','registry_url','model_inventory_url')

def load_config(path):
    cfg=json.loads(Path(path).read_text())
    if any(not cfg.get(k) for k in REQUIRED):raise ValueError('explicit model/compute/client/identity/command configuration required')
    if not isinstance(cfg['command'],list) or any(not isinstance(x,str) for x in cfg['command']):raise ValueError('command must be argv, never shell text')
    if not cfg['model'].get('digest'):raise ValueError('model digest pin required')
    if cfg.get('execution_path') not in ('native-service','rail-service'):raise ValueError('unsupported execution path')
    return cfg

def public_config(cfg):
    return {k:v for k,v in cfg.items() if k not in ('token_file','environment','profile_template')}

def preflight(cfg):
    import urllib.request
    p=Path(cfg['token_file']).expanduser()
    if p.stat().st_mode & 0o077:raise ValueError('token file must be private')
    bearer=p.read_text().strip()
    if not bearer or any(c.isspace() for c in bearer):raise ValueError('explicit bearer file required')
    with urllib.request.urlopen(urllib.request.Request(cfg['registry_url'].rstrip('/')+'/auth/verify',headers={'Authorization':'Bearer '+bearer}),timeout=10) as f:who=json.load(f)
    if any(who.get(k)!=v for k,v in cfg['identity'].items()):raise ValueError('requester identity mismatch')
    with urllib.request.urlopen(cfg['model_inventory_url'],timeout=10) as f:models=json.load(f)['models']
    if not any(m['name']==cfg['model']['name'] and m['digest']==cfg['model']['digest'] for m in models):raise ValueError('installed model digest mismatch')
    profile=Path(cfg['profile_template']).expanduser()/'config.yaml'
    if cfg['client'].get('kind')=='hermes':
        try: settings=json.loads(profile.read_text())
        except ValueError:
            import hermes_yaml
            settings=hermes_yaml.safe_load(profile.read_text())
        if settings['model']['default']!=cfg['model']['name'] or settings['model'].get('context_length')!=cfg['model'].get('context_length'):
            raise ValueError('Hermes profile does not match requested model/context')
    return bearer,{'identity':cfg['identity'],'model_digest_verified':True,'profile_model_verified':cfg['client'].get('kind')=='hermes'}

def profile_sources(cfg):
    template=Path(cfg['profile_template']).expanduser()
    files={}
    for name in ('config.yaml','SOUL.md'):
        p=template/name
        if p.exists():files[name]=p
    if 'config.yaml' not in files:raise ValueError('profile config missing')
    for p in (template/'plugins').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':files[str(p.relative_to(template))]=p
    return files

def prepare(directory,cfg,observed):
    sources=profile_sources(cfg)
    snapshot={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in sources.items()}
    # Bind environment without values in reports (it may contain credential references).
    public=public_config(cfg)
    binding=digest({'configuration':public,'environment':cfg['environment'],'profile_hashes':snapshot,'tasks':TASKS})
    if (directory/'state.json').exists():
        state=json.loads((directory/'state.json').read_text())
        if state['binding']!=binding:raise ValueError('configuration/profile/fixtures changed; resume refused')
        return state,sources
    manifest={'schema':SCHEMA,'fixture_revision':REVISION,'fixture_hash':digest(TASKS),
              'configuration':public,'profile_hashes':snapshot,'observed':observed,
              'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
              'assistance_policy':'No operator repair of submitted task; interventions invalidate unattended success',
              'limits':['Three small useful jobs, not a broad model ranking','Agent elapsed time includes queue/tool work',
                        'No memory peak, token efficiency or health score inferred from task pass','Qualitative handbacks require human review']}
    try:manifest['harness_commit']=subprocess.check_output(['git','-C',str(Path(__file__).parent),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):manifest['harness_commit']=None
    write_json(directory/'manifest.json',manifest)
    state={'schema':SCHEMA,'run_id':str(uuid.uuid4()),'binding':binding,'state':'prepared','tasks':{}}
    write_json(directory/'state.json',state)
    return state,sources

def run(cfg,directory,*,verified=None):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (directory/'observer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        bearer,observed=verified if verified is not None else preflight(cfg)
        state,sources=prepare(directory,cfg,observed)
        for name in cfg.get('tasks',list(TASKS)):
            if name not in TASKS:raise ValueError('unknown fixture')
            if name in state['tasks']:
                row=state['tasks'][name]
                if row['state']=='passed':continue
                # Even launch_intent may mean a running child. Never relaunch on uncertainty.
                return state
            task=TASKS[name];root=directory/name;work=root/'workspace';profile=root/'profile'
            work.mkdir(parents=True);profile.mkdir()
            for rel,text in task['files'].items():(work/rel).write_text(text)
            request=work/'request.txt';request.write_text(task['prompt'])
            for rel,p in sources.items():
                dest=profile/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
            substitutions={'workspace':str(work),'request':str(request),'profile':str(profile)}
            def render(text):
                for key,value in substitutions.items():text=text.replace('{'+key+'}',value)
                return text
            argv=[render(s) for s in cfg['command']]
            env={k:v for k,v in os.environ.items() if k not in ('SC_TOKEN','SC_TOKEN_FILE','OPENAI_API_KEY','SAMCLOUD_QUEUE_API_KEY','HERMES_HOME')}
            env.update({k:render(v) for k,v in cfg['environment'].items()})
            for key in cfg.get('token_environment',[]):env[key]=bearer
            row={'state':'launch_intent','argv':argv,'workspace':str(work),'started_at':time.time(),'operator_interventions':0}
            state['tasks'][name]=row;state['state']='running';write_json(directory/'state.json',state)
            with (root/'agent.log').open('ab') as log:
                child=subprocess.Popen(argv,cwd=work,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
                row.update(state='running',pid=child.pid);write_json(directory/'state.json',state)
                rc=child.wait()  # no inference timeout or automatic replacement
            row['returncode']=rc;row['elapsed_s']=time.time()-row['started_at']
            result=verify(name,work);result['unattended_process_completion']=rc==0
            row['result']=result
            row['artifact_hashes']={str(p.relative_to(work)):hashlib.sha256(p.read_bytes()).hexdigest() for p in work.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.is_symlink()}
            row['state']='passed' if rc==0 and result['state']=='passed' else 'failed'
            if row['state']=='failed':
                state['state']='failed';write_json(directory/'state.json',state);return state
            write_json(directory/'state.json',state)
        state['state']='complete';write_json(directory/'state.json',state);return state
