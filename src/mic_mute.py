"""Detects and clears an OS-level microphone mute before a recording session starts, and
reads (without changing) the mute state during one. Uses pycaw (a Python wrapper around
the Windows Core Audio API) to query and toggle the mute flag on the default capture
device - the same flag Windows' own volume mixer/tray icon controls.

Deliberately narrow in scope: this only sees the OS-level mute toggle on the default
input device. It can't see (and can't do anything about):
  - a physical hardware mute switch/button on some laptops
  - a mute set inside a specific app (Zoom, Teams, etc.) rather than at the OS level
  - the "system audio" recording source, which doesn't use a microphone at all

Windows-only, and optional - if pycaw isn't installed or the query fails for any reason,
callers get None back and should treat it the same as "couldn't determine, skip silently"
(matches how the rest of this app's preflight checks degrade)."""
import sys


def _default_mic_volume():
    """The default microphone's IAudioEndpointVolume interface, or None if unavailable."""
    if sys.platform != "win32":
        return None
    try:
        # comtypes calls CoInitializeEx() as a side effect of its own import, defaulting to
        # COINIT_APARTMENTTHREADED (STA) unless sys.coinit_flags says otherwise. This app's
        # audio capture (soundcard) already initializes COM as COINIT_MULTITHREADED (MTA) on
        # this thread by the time this ever runs - a second, conflicting CoInitializeEx call
        # raises "Cannot change thread mode after it is set" (RPC_E_CHANGED_MODE), which was
        # confirmed live: it silently broke every call to this function in the actual app,
        # since the failure happens inside this try/except and gets swallowed as "unavailable".
        # Setting this before comtypes' first import anywhere in the process matches the mode
        # soundcard already set, so its own CoInitializeEx call becomes a harmless no-op.
        if not hasattr(sys, "coinit_flags"):
            sys.coinit_flags = 0  # COINIT_MULTITHREADED
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        import comtypes

        mic = AudioUtilities.GetMicrophone()
        if mic is None:
            return None
        endpoint = mic.Activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None)
        return endpoint.QueryInterface(IAudioEndpointVolume)
    except Exception:
        return None


def check_and_unmute() -> str | None:
    """Checks the default microphone's mute state and clears it if muted. Returns a
    human-readable status line to print, or None if the check couldn't run at all
    (non-Windows, pycaw not installed, no default mic, COM failure, etc.) - callers
    should treat None as "nothing to report" rather than an error."""
    volume = _default_mic_volume()
    if volume is None:
        return None
    try:
        if not volume.GetMute():
            return None  # already unmuted - nothing worth printing
        volume.SetMute(0, None)
        return "unmuted"
    except Exception:
        return None


def is_muted() -> bool | None:
    """Read-only: whether Windows has the default microphone muted, or None if that can't be
    determined. Never changes the setting - it's used mid-recording to explain a lost
    signal, where silently unmuting a mic someone muted on purpose would be wrong."""
    volume = _default_mic_volume()
    if volume is None:
        return None
    try:
        return bool(volume.GetMute())
    except Exception:
        return None
