"""Open the existing interactive recorder from the local diagnostics dashboard."""
import os
from pathlib import Path
import subprocess
import threading
import time
import psutil
import telemetry

ROOT=Path(__file__).resolve().parents[1]
_lock=threading.Lock()
_child=None

def _own_process_ids():
    # The dashboard itself can be served from `python src/main.py` (menu option 4 or
    # --dashboard), and on Windows the venv python.exe is a launcher parent with the same
    # command line. Neither is a recorder, so they must not disable Start Notes forever.
    ids={os.getpid()}
    try:ids.update(p.pid for p in psutil.Process().parents())
    except psutil.Error:pass
    return ids

def recorder_state():
    if _child is not None and _child.poll() is None:
        return {'running':True,'pid':_child.pid}
    expected=(ROOT/'src/main.py').resolve()
    own=_own_process_ids()
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            if proc.info['pid'] in own:continue
            if not (proc.info['name'] or '').lower().startswith(('python','pythonw')):continue
            args=proc.info['cmdline'] or []
            for arg in args[1:]:
                if Path(arg).name.lower()!='main.py':continue
                candidate=Path(arg)
                if not candidate.is_absolute():candidate=Path(proc.cwd())/candidate
                if candidate.resolve()==expected:return {'running':True,'pid':proc.info['pid']}
        except (psutil.Error,OSError,ValueError):continue
    status=telemetry.read_status()
    if status.get('active') and time.time()-status.get('updated_at',0)<30:
        return {'running':True,'pid':None}
    return {'running':False,'pid':None}

def start_recorder():
    global _child
    with _lock:
        current=recorder_state()
        if current['running']:
            return {'ok':True,'already_running':True,'message':'Notes is already open. Use its startup window or the recording controls below.'}
        python=ROOT/'venv/Scripts/python.exe';entry=ROOT/'src/main.py'
        if not python.is_file() or not entry.is_file():raise FileNotFoundError('Recorder files are missing. Check the lecture-notes Python environment.')
        if os.name!='nt':raise RuntimeError('Opening the interactive Notes window currently requires Windows.')
        _child=subprocess.Popen([str(python),str(entry)],cwd=str(ROOT),creationflags=subprocess.CREATE_NEW_CONSOLE)
        return {'ok':True,'already_running':False,'pid':_child.pid,
                'message':'Notes startup opened. Choose your class, audio source, and formatting there; live status will appear here when recording begins.'}
