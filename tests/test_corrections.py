import json
import sys
import time
import threading
import http.client
from pathlib import Path
from http.server import ThreadingHTTPServer

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import corrections as c
import transcripts
import vocab
import telemetry
import storage
import dashboard
import search

SID='BIO_101_20260911_120000'


@pytest.fixture
def library(tmp_path,monkeypatch):
    state,notes=tmp_path/'state',tmp_path/'notes'
    state.mkdir();notes.mkdir()
    monkeypatch.setattr(c,'STATE_DIR',state)
    monkeypatch.setattr(c,'NOTES_DIR',notes)
    monkeypatch.setattr(vocab,'VOCAB_PATH',tmp_path/'vocab.json')
    monkeypatch.setattr(telemetry,'read_status',lambda:{})
    import docx_export
    monkeypatch.setattr(docx_export,'rebuild',lambda *args:None)
    source='\n'.join(json.dumps(s) for s in [
        {'start':0,'end':4,'text':'The mitocondria produce energy.'},
        {'start':4,'end':8,'text':'Membranes control transport.'}])+'\n'
    (state/(SID+'_segments.jsonl')).write_text(source)
    (state/(SID+'_raw.txt')).write_text('[12:00:00] The mitocondria produce energy.\n')
    (state/(SID+'.wav')).write_bytes(b'unchanged source audio')
    text=storage.update_session('# Biology (BIO 101)\n','earlier','## Yesterday\n\n### Keep\nUnrelated prior lecture.\n')
    text=storage.update_session(text,SID,'## Today\n\n### Energy\nOld energy wording.\n\n### Membranes\nKeep this topic exactly.\n')
    (notes/'BIO_101.md').write_text(text)
    with c._jobs_lock:c._jobs.clear()
    return state,notes


def payload():
    data=c.workspace(SID)
    section=next(s for s in data['sections'] if s['title']=='Energy')
    return {'session_id':SID,'section_id':section['id'],'notes_revision':data['notes_revision'],
            'transcript_revision':data['transcript_revision'],'segment_indices':[0],'mode':'heuristic'}


def ready(monkeypatch):
    monkeypatch.setattr(c,'generate',lambda *args:('### Energy\n\n- Corrected energy wording.\n','test formatter'))
    job=c.start_regeneration(payload())['job_id']
    deadline=time.monotonic()+3
    while c.get_job(job)['status']=='running' and time.monotonic()<deadline:time.sleep(.01)
    assert c.get_job(job)['status']=='ready'
    return job


def test_correction_preserves_source_and_remembers_term(library):
    state,_=library
    before=(state/(SID+'_segments.jsonl')).read_bytes()
    data=c.workspace(SID)
    c.save_correction({'session_id':SID,'transcript_revision':data['transcript_revision'],
                       'edits':[{'index':0,'text':'The mitochondria produce energy.'}],
                       'glossary_term':'mitochondria'})
    assert (state/(SID+'_segments.jsonl')).read_bytes()==before
    assert (state/(SID+'.wav')).read_bytes()==b'unchanged source audio'
    data=c.workspace(SID)
    assert data['segments'][0]['original']=='The mitocondria produce energy.'
    assert data['segments'][0]['text']=='The mitochondria produce energy.'
    assert 'mitochondria' in vocab.terms_for_class('BIO 101')


def test_apply_only_selected_section_and_preserve_revision(library,monkeypatch):
    _,notes=library
    before=(notes/'BIO_101.md').read_text()
    p=payload();section=next(s for s in c.workspace(SID)['sections'] if s['id']==p['section_id'])
    job=ready(monkeypatch)
    assert (notes/'BIO_101.md').read_text()==before  # preview is read-only
    c.apply_preview({'job_id':job})
    after=(notes/'BIO_101.md').read_text()
    assert after.startswith(before[:section['start']])
    assert after.endswith(before[section['end']:])
    assert 'Unrelated prior lecture.' in after and 'Keep this topic exactly.' in after
    assert next((notes/'.revisions'/'BIO_101.md').glob('*.txt')).read_text()==before
    with pytest.raises(storage.RevisionConflict):c.apply_preview({'job_id':job})


def test_stale_notes_cannot_be_overwritten(library,monkeypatch):
    _,notes=library
    job=ready(monkeypatch)
    path=notes/'BIO_101.md';path.write_text(path.read_text()+'Newer edit\n')
    with pytest.raises(storage.RevisionConflict):c.apply_preview({'job_id':job})
    assert path.read_text().endswith('Newer edit\n')


def test_stale_transcript_rejected_and_preview_invalidated(library,monkeypatch):
    job=ready(monkeypatch);data=c.workspace(SID)
    change={'session_id':SID,'transcript_revision':data['transcript_revision'],
            'edits':[{'index':0,'text':'Reviewed wording'}]}
    c.save_correction(change)
    with pytest.raises(storage.RevisionConflict):c.save_correction(change)
    with pytest.raises(storage.RevisionConflict):c.apply_preview({'job_id':job})


def test_legacy_source_works_without_invented_timing(library):
    state,_=library;(state/(SID+'_segments.jsonl')).unlink()
    data=c.workspace(SID)
    assert data['segments'][0]['start'] is None and not data['timed']
    c.save_correction({'session_id':SID,'transcript_revision':data['transcript_revision'],
                       'edits':[{'index':0,'text':'Corrected legacy text'}]})
    assert c.workspace(SID)['segments'][0]['text']=='Corrected legacy text'


def test_search_uses_saved_correction(library,monkeypatch):
    state,_=library;data=c.workspace(SID)
    c.save_correction({'session_id':SID,'transcript_revision':data['transcript_revision'],
                       'edits':[{'index':0,'text':'Mitochondria make ATP.'}]})
    monkeypatch.setattr(search,'STATE_DIR',state)
    hits=search.search_transcripts('ATP')
    assert len(hits)==1 and hits[0].audio_seconds==0


def test_running_recorder_and_wrong_session_rejected(library,monkeypatch):
    p=payload()
    p['section_id']=next(s for s in c.workspace(SID)['sections'] if s['title']=='Keep')['id']
    with pytest.raises(ValueError):c.start_regeneration(p)
    monkeypatch.setattr(telemetry,'read_status',lambda:{'active':True,'class_code':'BIO 101','updated_at':time.time()})
    with pytest.raises(storage.RevisionConflict):c.start_regeneration(payload())


def test_no_extra_heading_or_marker_in_applied_text(library,monkeypatch):
    job=ready(monkeypatch)
    for text in ('### Other\nChanged','### Energy\n## Another section\nBad','### Energy\n<!-- session:x -->'):
        with pytest.raises(ValueError):c.apply_preview({'job_id':job,'text':text})


def test_source_revision_change_does_not_misapply_overlay(library):
    state,_=library;data=c.workspace(SID)
    c.save_correction({'session_id':SID,'transcript_revision':data['transcript_revision'],
                       'edits':[{'index':0,'text':'Reviewed'}]})
    with (state/(SID+'_segments.jsonl')).open('a') as f:f.write('{}\n')
    with pytest.raises(ValueError,match='source transcript changed'):c.workspace(SID)


def test_http_write_origin_and_validation(library):
    server=ThreadingHTTPServer(('127.0.0.1',0),dashboard.Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
        conn.request('POST','/api/corrections/save','{}',{'Content-Type':'application/json','Origin':'https://elsewhere.example'})
        response=conn.getresponse();assert response.status==403;response.read()
        conn.request('POST','/api/corrections/save','{}',{'Content-Type':'application/json','Host':f'elsewhere.example:{server.server_port}','Origin':f'http://elsewhere.example:{server.server_port}'})
        response=conn.getresponse();assert response.status==403;response.read()
        conn.request('GET','/api/corrections/session?session=../secret')
        response=conn.getresponse();assert response.status==409;response.read()
        conn.request('POST','/api/corrections/save','[]',{'Content-Type':'application/json'})
        response=conn.getresponse();assert response.status==400;response.read();conn.close()
    finally:server.shutdown();server.server_close()


def test_optimistic_file_write_is_atomic(tmp_path):
    path=tmp_path/'notes.md';path.write_text('initial')
    storage.atomic_write(path,'new',expected='initial')
    with pytest.raises(storage.RevisionConflict):storage.atomic_write(path,'stale',expected='initial')
    assert path.read_text()=='new'


def test_recovery_keeps_reviewed_words_instead_of_retranscribing(library,monkeypatch):
    import main
    from types import SimpleNamespace
    state,notes=library
    data=c.workspace(SID)
    c.save_correction({'session_id':SID,'transcript_revision':data['transcript_revision'],
                       'edits':[{'index':0,'text':'Reviewed mitochondria wording.'}]})
    monkeypatch.setattr(main.notes,'NOTES_DIR',notes)
    monkeypatch.setattr(main.sched,'get_class_by_code',lambda code:{'title':'Biology'})
    monkeypatch.setattr(main.diarize,'available',lambda:False)
    def forbidden():raise AssertionError('Reviewed text must not be re-transcribed')
    monkeypatch.setattr(main.transcribe,'get_model',forbidden)
    captured=[]
    def save(code,title,text,*args,**kwargs):
        captured.append(text)
        return notes/'BIO_101.md','local'
    monkeypatch.setattr(main.notes,'format_and_save',save)
    main.run_resume(str(state/(SID+'.wav')),SimpleNamespace(formatting='heuristic'))
    assert 'Reviewed mitochondria wording.' in captured[0]


def test_generation_fallback_uses_only_selected_excerpt():
    preview,method=c.generate('### Energy\nold content','Corrected source sentence.','heuristic')
    assert preview=='### Energy\n\n- Corrected source sentence.\n'
    assert 'no model' in method


def test_failed_generation_does_not_write_notes(library,monkeypatch):
    _,notes=library;before=(notes/'BIO_101.md').read_bytes()
    def fail(*args):raise RuntimeError('formatter offline')
    monkeypatch.setattr(c,'generate',fail)
    job=c.start_regeneration(payload())['job_id']
    deadline=time.monotonic()+3
    while c.get_job(job)['status']=='running' and time.monotonic()<deadline:time.sleep(.01)
    assert c.get_job(job)['status']=='failed'
    assert (notes/'BIO_101.md').read_bytes()==before
