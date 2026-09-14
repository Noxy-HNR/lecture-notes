import http.client
import json
import sys
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer
from unittest.mock import Mock
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import recorder_launcher as launcher
import dashboard

@pytest.fixture
def clean(monkeypatch):
    monkeypatch.setattr(launcher,'_child',None)
    monkeypatch.setattr(launcher.psutil,'process_iter',lambda *a:[])
    monkeypatch.setattr(launcher.telemetry,'read_status',lambda:{})

def test_start_opens_existing_interactive_entrypoint_only_once(clean,monkeypatch):
    child=Mock(pid=1234);child.poll.return_value=None
    popen=Mock(return_value=child);monkeypatch.setattr(launcher.subprocess,'Popen',popen)
    first=launcher.start_recorder();second=launcher.start_recorder()
    assert first['ok'] and not first['already_running'] and second['already_running']
    assert popen.call_count==1
    args,options=popen.call_args
    assert args[0]==[str(launcher.ROOT/'venv/Scripts/python.exe'),str(launcher.ROOT/'src/main.py')]
    assert options['creationflags']==launcher.subprocess.CREATE_NEW_CONSOLE

def test_external_recorder_is_detected_after_dashboard_restart(clean,monkeypatch):
    process=Mock();process.info=dict(pid=99,name='python.exe',cmdline=['python','src/main.py'])
    process.cwd.return_value=str(launcher.ROOT)
    monkeypatch.setattr(launcher.psutil,'process_iter',lambda *a:[process])
    assert launcher.recorder_state()=={'running':True,'pid':99}

def test_dashboard_served_from_main_py_is_not_mistaken_for_a_recorder(clean,monkeypatch):
    """Menu option 4 and --dashboard serve the dashboard from `python src/main.py`, whose
    parent is the venv python.exe launcher with the same command line. Neither records."""
    import os
    def proc(pid):
        p=Mock();p.info=dict(pid=pid,name='python.exe',cmdline=['python','src/main.py'])
        p.cwd.return_value=str(launcher.ROOT);return p
    monkeypatch.setattr(launcher.psutil,'process_iter',lambda *a:[proc(os.getpid()),proc(4242)])
    monkeypatch.setattr(launcher.psutil,'Process',lambda *a:Mock(parents=lambda:[Mock(pid=4242)]))
    assert launcher.recorder_state()=={'running':False,'pid':None}

def test_command_api_rejects_non_local_host(clean,monkeypatch):
    """DNS rebinding: a page served from attacker.example:PORT that re-resolves to 127.0.0.1
    is same-origin with itself, so an Origin == Host check alone would let it stop a recording."""
    start=Mock();monkeypatch.setattr(launcher,'start_recorder',start)
    send=Mock(return_value=True);monkeypatch.setattr(dashboard.telemetry,'send_command',send)
    server=ThreadingHTTPServer(('127.0.0.1',0),dashboard.Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    host=f'attacker.example:{server.server_port}'
    try:
        for action in ('stop','save_now','start'):
            conn=http.client.HTTPConnection('127.0.0.1',server.server_port)
            conn.request('POST','/api/command',json.dumps({'action':action}),
                         {'Content-Type':'application/json','X-Notes-Dashboard':'1','Host':host,'Origin':'http://'+host})
            response=conn.getresponse();response.read();conn.close()
            assert response.status==403
        send.assert_not_called();start.assert_not_called()
    finally:server.shutdown();server.server_close();thread.join()

def test_finished_window_can_be_started_again(clean,monkeypatch):
    child=Mock();child.poll.return_value=1;monkeypatch.setattr(launcher,'_child',child)
    assert not launcher.recorder_state()['running']

def test_command_api_success_error_and_origin_protection(clean,monkeypatch):
    start=Mock(return_value={'ok':True,'message':'Startup opened'})
    monkeypatch.setattr(launcher,'start_recorder',start)
    save=Mock(return_value=True);monkeypatch.setattr(dashboard.telemetry,'send_command',save)
    server=ThreadingHTTPServer(('127.0.0.1',0),dashboard.Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    def post(data,headers=None):
        conn=http.client.HTTPConnection('127.0.0.1',server.server_port)
        conn.request('POST','/api/command',json.dumps(data),headers or {})
        response=conn.getresponse();result=response.status,json.loads(response.read());conn.close();return result
    headers={'Content-Type':'application/json','X-Notes-Dashboard':'1'}
    try:
        assert post({'action':'start'})[0]==403
        assert post({'action':'start'},{**headers,'Origin':'https://example.com'})[0]==403
        assert start.call_count==0
        assert post({'action':'start'},headers)==(200,{'ok':True,'message':'Startup opened'})
        assert post({'action':'stop'},headers)[1]['ok'];save.assert_called_with('stop')
        assert post([],headers)[0]==400
        start.side_effect=OSError('Could not start Python')
        assert post({'action':'start'},headers)==(500,{'ok':False,'error':'Could not start Python'})
    finally:server.shutdown();server.server_close();thread.join()
