"""One locally planned public primary experiment through existing stage functions.

This release-preparation driver refuses existing evidence and records all stage
outputs. It performs no outcome-based filtering, retries, or replacement runs.
For ordinary fresh experiments use ``python -m kahm_demo reproduce`` instead.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import time

PROJECT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(PROJECT))
from kahm_demo import __version__
from kahm_demo._util import canonical_json_bytes, read_json, sha256_file
from kahm_demo.audit_workflow import audit_certificate_family
from kahm_demo.calibration_protocol import CALIBRATION_PROTOCOL_SHA256, load_calibration_protocol
from kahm_demo.certificate_workflow import certify_frozen_family
from kahm_demo.contract import CONTRACT_SHA256, validate_contract
from kahm_demo.datasets import generate_bundle
from kahm_demo.encoder_workflow import fit_development_encoders
from kahm_demo.forecast_workflow import fit_development_forecasters, forecast_runtime_record
from kahm_demo.freeze import freeze_forecasters
from kahm_demo.sampling import stream_descriptor
from kahm_demo.reproduce import _ordered_rows

SEED = 20261110
RUN_ID = 'public-v0.1.0'
PATHS = {name: f'{name}/primary' for name in ('data','encoders','models','frozen','certificates','audits')}
REPORTS = PROJECT / 'reports'
STAGES = ['generate_data','fit_encoders','fit_forecasters','freeze_models','certify','audit']


def record(path, value):
    value = dict(value)
    value['record_sha256'] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    with path.open('xb') as handle:
        handle.write(canonical_json_bytes(value))
    return sha256_file(path)


def sources():
    return {x.relative_to(PROJECT).as_posix(): sha256_file(x)
            for x in sorted((PROJECT/'kahm_demo').iterdir()) if x.suffix in ('.py','.json')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


validate_contract(); load_calibration_protocol()
for relative in PATHS.values():
    path=PROJECT/relative
    require(not path.exists() and not path.is_symlink(), f'Existing evidence is never replaced: {relative}')
REPORTS.mkdir(exist_ok=True)
plan_path=REPORTS/'public_primary_plan.json'
require(not plan_path.exists(), 'This primary experiment has already been planned; no rerun or resume.')
source_identities=sources()
driver_hash=sha256_file(Path(__file__))
plan={
    'schema':'kahm.public_primary_plan.v1','status':'planned','package_version':__version__,
    'local_recorded_at_utc':datetime.now(timezone.utc).isoformat(),
    'run_id':RUN_ID,'master_seed_integer':str(SEED),
    'training_count':4096,'validation_count':1024,'calibration_count':4096,
    'encoder_caps':[1,2],'forecast_search_count':225,
    'candidate_ids':['raw','kahm_mem','kahm_local'],'stage_order':STAGES,'paths':PATHS,
    'contract_sha256':CONTRACT_SHA256,'calibration_protocol_sha256':CALIBRATION_PROTOCOL_SHA256,
    'streams':{r:stream_descriptor(SEED,r,RUN_ID) for r in ('training','validation','calibration')},
    'runtime':forecast_runtime_record(),'source_sha256':source_identities,
    'driver_sha256':driver_hash,
    'policy':{'automatic_retries':False,'resume':False,'overwrite':False,'filter_unfavorable_outcomes':False,
              'calibration_after_freeze_only':True,'population_audit_after_verified_certificate_only':True,
              'public_preregistration_claimed':False,'prng_proves_iid_premise':False,
              'tutorial_anchors_assigned_after_evidence_issuance':True}
}
plan_hash=record(plan_path,plan)
completed=[]; outputs={}; stage='commit_plan'; start=time.perf_counter()

def perform(name, function, *args, expected_status, verified=False, **kwargs):
    global stage
    stage=name
    require(sha256_file(plan_path)==plan_hash,'Plan changed after commitment')
    require(sources()==source_identities,'Package sources changed during primary execution')
    require(sha256_file(Path(__file__))==driver_hash,'Driver changed after commitment')
    print(name,flush=True)
    output=function(*args,**kwargs)
    outputs[name]=output
    record(REPORTS/f'public_primary_{name}.json',{'schema':'kahm.public_primary_stage_result.v1',
        'stage':name,'plan_sha256':plan_hash,'result':output})
    require(output.get('status')==expected_status,f'{name} did not complete: {output.get("status")}')
    if verified: require(isinstance(output.get('verification'), dict) and output['verification'].get('status') == 'passed', f'{name} did not verify its artifact')
    completed.append(name)
    return output

try:
    d,e,m,f,c,a=(PROJECT/PATHS[n] for n in ('data','encoders','models','frozen','certificates','audits'))
    perform('generate_data',generate_bundle,d,master_seed=SEED,run_id=RUN_ID,
            training_count=4096,validation_count=1024,expected_status='passed')
    result=perform('fit_encoders',fit_development_encoders,d,e,caps=(1,2),expected_status='completed')
    require(result.get('bank_order')==['cap_1','cap_2'],'Encoder menu changed')
    result=perform('fit_forecasters',fit_development_forecasters,d,e,m,expected_status='completed')
    require(result.get('search_count')==225,'Forecast menu changed')
    frozen=perform('freeze_models',freeze_forecasters,d,e,m,f,expected_status='frozen',verified=True)
    freeze_hash=sha256_file(f/'freeze.json')
    require(frozen.get('freeze_record_sha256')==freeze_hash,'Freeze return does not bind the saved artifact')
    result=perform('certify',certify_frozen_family,d,f/'freeze.json',c,expected_status='assessment_complete',verified=True)
    require(result.get('calibration_count')==4096 and result.get('calibration_action')=='materialize_reserved_stream',
            'Calibration count or action changed')
    certified=result
    certificate_hash=sha256_file(c/'certificate.json')
    require(certified.get('certificate_sha256')==certificate_hash,'Certificate return does not bind the saved artifact')
    _ordered_rows(certified.get('candidates'),'status')
    result=perform('audit',audit_certificate_family,c/'certificate.json',dataset_bundle=d,
                   freeze_path=f/'freeze.json',output=a,expected_status='audit_complete',verified=True)
    audit_hash=sha256_file(a/'audit.json')
    require(result.get('audit_sha256')==audit_hash,'Audit return does not bind the saved artifact')
    require(sha256_file(plan_path)==plan_hash,'Plan changed during final audit')
    require(sha256_file(Path(__file__))==driver_hash,'Driver changed during final audit')
    require(sources()==source_identities,'Sources changed during public execution')
    require(sha256_file(f/'freeze.json')==freeze_hash and sha256_file(c/'certificate.json')==certificate_hash,
            'An issued artifact changed after its verified stage')
    certificate=read_json(c/'certificate.json'); audit=read_json(a/'audit.json')
    cert_rows=_ordered_rows(certificate.get('candidates'),'status')
    audit_rows=_ordered_rows(audit.get('candidates'),'certificate_status')
    require(canonical_json_bytes(cert_rows)==canonical_json_bytes(certified['candidates'])
            and canonical_json_bytes(audit_rows)==canonical_json_bytes(result.get('candidates'))
            and canonical_json_bytes(audit.get('summary'))==canonical_json_bytes(result.get('summary'))
            and all(before['status']==after['certificate_status']
                    for before,after in zip(cert_rows,audit_rows,strict=True)),
            'Returned and saved family results disagree')
    artifacts={name:{'path':str(path.relative_to(PROJECT)),'sha256':sha256_file(path)} for name,path in
               [('freeze',f/'freeze.json'),('certificate',c/'certificate.json'),('audit',a/'audit.json')]}
    final={'schema':'kahm.public_primary_execution.v1','status':'pipeline_complete','pipeline_complete':True,
           'plan_sha256':plan_hash,'run_id':RUN_ID,'master_seed_integer':str(SEED),'completed_stages':completed,
           'artifacts':artifacts,'candidates':result['candidates'],'audit_summary':result['summary'],
           'automatic_retries_performed':False,'unfavorable_outcomes_filtered':False,
           'elapsed_seconds':time.perf_counter()-start,
           'interpretation':'One locally planned execution. Every candidate outcome is retained. Statistical claims retain the conditional iid premise.'}
    record(REPORTS/'public_primary_execution.json',final)
    print(json.dumps({'status':final['status'],'artifacts':artifacts,
                      'outcomes':[{k:r.get(k) for k in ('candidate_id','certificate_status','realized_coverage')} for r in result['candidates']]}),flush=True)
except BaseException as error:
    record(REPORTS/'public_primary_failure.json',{'schema':'kahm.public_primary_failure.v1','status':'pipeline_failed',
        'phase':stage,'completed_stages':completed,'plan_sha256':plan_hash,'run_id':RUN_ID,
        'error_type':type(error).__name__,'error_message':str(error),'partial_evidence_retained':True,
        'automatic_retries_performed':False,'elapsed_seconds':time.perf_counter()-start})
    raise
