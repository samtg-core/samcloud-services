"""Configured agent jobs, held-out checks, immutable comparison records.

No physical allocator, queue transport, rail fork or automatic replacement job.
Optional case-end grant return releases only the exact grants a passed case's own
profile bound, through the supported release endpoint, before the next dispatch.
"""
import fcntl
import hashlib
import json
import inspect
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from .core import SCHEMA,digest,write_json
from . import tasks
from .tasks import REVISION,TASKS

REQUIRED=('model','compute','client','identity','service_id','resource_id','command','environment','profile_template','token_file','registry_url','model_inventory_url')

def load_config(path):
    cfg=json.loads(Path(path).read_text())
    if any(not cfg.get(k) for k in REQUIRED):raise ValueError('explicit model/compute/client/identity/command configuration required')
    if not isinstance(cfg['command'],list) or any(not isinstance(x,str) for x in cfg['command']):raise ValueError('command must be argv, never shell text')
    if not cfg['model'].get('digest'):raise ValueError('model digest pin required')
    if cfg.get('execution_path') != 'native-service':raise ValueError('unsupported execution path')
    if 'grant_return' in cfg:grant_return_config(cfg)
    return cfg

class GrantReturnUnconfirmed(Exception):
    """A passed case's grant could not be shown returned; later dispatch stops."""

RETURN_ACK={'queued':'cancelled','granted':'released'}
BINDING_FIELDS={'id':'grant_id','owner':'owner','client_key':'key','resource':'resource','model':'model'}

def grant_return_config(cfg):
    g=cfg['grant_return']
    if not isinstance(g,dict) or set(g)!={'identity_store','gateway_url'}:raise ValueError('grant_return needs exactly identity_store and gateway_url')
    store=Path(g['identity_store'])
    if store.is_absolute() or '..' in store.parts or not store.parts:raise ValueError('grant_return identity_store must be relative to the case profile')
    url=urllib.parse.urlsplit(g['gateway_url'])
    if url.scheme not in ('http','https') or not url.netloc:raise ValueError('grant_return gateway_url must be an http(s) URL')
    if not cfg.get('identity',{}).get('username'):raise ValueError('grant_return requires the requester username')
    return g

def case_grant_bindings(profile,store):
    """Exact grants this case's own profile bound; no discovery outside it."""
    base=Path(profile).resolve();path=Path(profile)/store
    if not path.is_file() or path.is_symlink():raise GrantReturnUnconfirmed('case identity store missing')
    real=path.resolve()
    # Source scope: the store must resolve inside this case's own profile.
    if real==base or base not in real.parents:raise GrantReturnUnconfirmed('case identity store resolves outside the case profile')
    db=sqlite3.connect(real.as_uri()+'?mode=ro',uri=True)
    try:rows=db.execute('SELECT key,grant_id,owner,resource,model,lifetime FROM grant_binding ORDER BY rowid').fetchall()
    except sqlite3.Error as error:raise GrantReturnUnconfirmed('case identity store has no grant binding ('+type(error).__name__+')')
    finally:db.close()
    if not rows:raise GrantReturnUnconfirmed('passed case bound no grant')
    return [dict(zip(('key','grant_id','owner','resource','model','lifetime'),row)) for row in rows]

def gateway(method,url,bearer):
    request=urllib.request.Request(url,data=b'' if method=='POST' else None,method=method,headers={'Authorization':'Bearer '+bearer})
    try:
        with urllib.request.urlopen(request,timeout=30) as f:return json.load(f)
    except urllib.error.HTTPError as error:
        raise GrantReturnUnconfirmed(method+' grant refused (HTTP '+str(error.code)+')')
    except (urllib.error.URLError,OSError,ValueError) as error:
        # A lost reply is unknown, not a failure: the same id is retried on resume.
        raise GrantReturnUnconfirmed(method+' grant reply unknown ('+type(error).__name__+')')

def observed_grant(reply,binding,username):
    if not isinstance(reply,dict):raise GrantReturnUnconfirmed('grant reply is not an object')
    for field,key in BINDING_FIELDS.items():
        if reply.get(field)!=binding[key]:raise GrantReturnUnconfirmed('grant '+field+' does not match the case binding')
    if binding['owner']!=username:raise GrantReturnUnconfirmed('bound grant owner is not the requester')
    return {k:reply.get(k) for k in ('id','owner','client_key','resource','model','state','release_requested','active_execution')}

def return_case_grants(cfg,bearer,row,profile,save):
    """Logical return of a passed case's exact idle grants; not proof of unloaded residency."""
    g=grant_return_config(cfg);username=cfg['identity']['username'];base=g['gateway_url'].rstrip('/')+'/inference/grants/'
    ret=row.get('grant_return')
    if ret is None or not ret.get('grants'):
        # No durable intent yet, or an unbound record: (re)read the exact, nonempty binding set.
        ret=row['grant_return']={'state':'intent','at':time.time(),'grants':[{'binding':b} for b in case_grant_bindings(profile,g['identity_store'])]}
        save()  # durable intent and exact ids before any release request
    ret['state']='intent';ret.pop('reason',None)
    for item in ret['grants']:
        if item.get('ack'):continue
        b=item['binding'];url=base+urllib.parse.quote(b['grant_id'],safe='')
        before=observed_grant(gateway('GET',url,bearer),b,username);item['before']=before
        if before['state'] in RETURN_ACK.values() and not before['active_execution']:
            item['ack']={**before,'observed':'terminal_without_new_request'};save();continue
        if before['state'] not in RETURN_ACK:raise GrantReturnUnconfirmed('grant state '+str(before['state'])+' is not returnable')
        if before['active_execution']:raise GrantReturnUnconfirmed('grant still has active work')
        item['release_requested_at']=time.time();save()
        after=observed_grant(gateway('POST',url+'/release',bearer),b,username)
        if after['release_requested'] not in (1,True) or after['active_execution'] or after['state']!=RETURN_ACK[before['state']]:
            item['reply']=after;save();raise GrantReturnUnconfirmed('release reply is not a completed return')
        item['ack']={**after,'observed':'release_reply'};save()
    if not ret['grants'] or not all(item.get('ack') for item in ret['grants']):raise GrantReturnUnconfirmed('return intent has no acknowledged grant')
    ret['state']='returned';save()

def settle_grant_return(cfg,bearer,state,directory,name,row):
    """True when later dispatch may proceed. Unconfirmed returns stop the cohort."""
    if 'grant_return' not in cfg or (row.get('grant_return') or {}).get('state')=='returned':return True
    save=lambda:write_json(directory/'state.json',state)
    try:
        return_case_grants(cfg,bearer,row,directory/name/'profile',save);return True
    except GrantReturnUnconfirmed as error:
        ret=row.get('grant_return')
        if ret and ret.get('grants'):ret.update(state='unconfirmed',reason=str(error))
        else:row['grant_return']={'state':'unbound','reason':str(error)}  # never an intent; resume re-reads the store
        state['state']='grant_return_unconfirmed';save();return False


def minimal_environment():
    return {k:os.environ[k] for k in ('PATH','LANG','LC_ALL','TMPDIR') if k in os.environ}

def source_hashes():
    result={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    result['active_checker']=hashlib.sha256(inspect.getsource(tasks.verify).encode()).hexdigest()
    return result

def redact_evidence(root,bearer):
    needle=bearer.encode();found=[]
    if not needle:raise ValueError('empty bearer cannot be scanned')
    for path in Path(root).rglob('*'):
        if path.is_file() and not path.is_symlink():
            data=path.read_bytes()
            if needle in data:
                found.append(str(path.relative_to(root)))
                # Keep the safe evidence only; raw secret-bearing bytes are not archived.
                with path.open('wb') as f:
                    f.write(data.replace(needle,b'[REDACTED]'));f.flush();os.fsync(f.fileno())
    return found

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
    hashes=source_hashes()
    binding=digest({'configuration':public,'environment':cfg['environment'],'profile_hashes':snapshot,'tasks':TASKS,'source_hashes':hashes})
    if (directory/'state.json').exists():
        state=json.loads((directory/'state.json').read_text())
        if state['binding']!=binding:raise ValueError('configuration/profile/fixtures changed; resume refused')
        return state,sources
    manifest={'schema':SCHEMA,'fixture_revision':REVISION,'fixture_hash':digest(TASKS),
              'configuration':public,'profile_hashes':snapshot,'observed':observed,
              'source_hashes':hashes,'provenance':'executed',
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
    if cfg.get('execution_path') != 'native-service':raise ValueError('rail adapter with delivery receipts is not implemented')
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (directory/'observer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        bearer,observed=verified if verified is not None else preflight(cfg)
        state,sources=prepare(directory,cfg,observed)
        for name in cfg.get('tasks',list(TASKS)):
            if name not in TASKS:raise ValueError('unknown fixture')
            if name in state['tasks']:
                row=state['tasks'][name]
                if row['state']=='passed':
                    # An outstanding return intent is settled BEFORE skipping the passed case.
                    if not settle_grant_return(cfg,bearer,state,directory,name,row):return state
                    continue
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
            env=minimal_environment()
            env.update({k:render(v) for k,v in cfg['environment'].items()})
            for key in cfg.get('token_environment',[]):env[key]=bearer
            row={'state':'launch_intent','argv':argv,'workspace':str(work),'started_at':time.time(),'operator_interventions':None}
            state['tasks'][name]=row;state['state']='running';write_json(directory/'state.json',state)
            with (root/'agent.log').open('ab') as log:
                child=subprocess.Popen(argv,cwd=work,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
                row.update(state='running',pid=child.pid);write_json(directory/'state.json',state)
                rc=child.wait()  # no inference timeout or automatic replacement
            row['returncode']=rc;row['elapsed_s']=time.time()-row['started_at']
            leaked=redact_evidence(root,bearer)
            result=({'state':'failed','failure':'secret_in_evidence','checks':{},'redacted_files':leaked}
                    if leaked else tasks.verify(name,work))
            result['process_completed']=rc==0
            result['unattended']=None
            # Generated checker code is untrusted evidence too. Scan after verification.
            leaked_after=redact_evidence(root,bearer)
            if leaked_after:
                result.update(state='failed',failure='secret_in_evidence',redacted_files=sorted(set(leaked+leaked_after)))
            row['result']=result
            row['artifact_hashes']={str(p.relative_to(work)):hashlib.sha256(p.read_bytes()).hexdigest() for p in work.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.is_symlink()}
            row['state']='passed' if rc==0 and result['state']=='passed' else 'failed'
            if row['state']=='failed':
                # Failed or interrupted work keeps its grant: retained debt, never a return.
                state['state']='failed';write_json(directory/'state.json',state);return state
            write_json(directory/'state.json',state)
            if not settle_grant_return(cfg,bearer,state,directory,name,row):return state
        state['state']='complete';write_json(directory/'state.json',state);return state
