"""Whether a word is correctly spelled English, from Windows' own spell checker.

The glossary fix-up uses this before changing a word: a real word such as "action" or
"dehydration" must never be swapped for a glossary term it happens to resemble ("cation",
"hydration"). The checker is a COM object, so each thread gets its own; answers are cached.
Off Windows, or if the checker can't be created, is_english_word() returns None and callers
make no fuzzy changes at all.
"""
import sys
import threading

_LANGUAGE = "en-US"
_local = threading.local()


def _interfaces():
    from ctypes import POINTER, c_int, c_ulong, c_wchar_p
    from comtypes import COMMETHOD, GUID, HRESULT, IUnknown

    class ISpellingError(IUnknown):
        _iid_ = GUID("{B7C82D61-FBE8-4B47-9B27-6C0D2E0DE0A3}")
        _methods_ = [
            COMMETHOD([], HRESULT, "get_StartIndex", (["out"], POINTER(c_ulong), "value")),
            COMMETHOD([], HRESULT, "get_Length", (["out"], POINTER(c_ulong), "value")),
            COMMETHOD([], HRESULT, "get_CorrectiveAction", (["out"], POINTER(c_int), "value")),
            COMMETHOD([], HRESULT, "get_Replacement", (["out"], POINTER(c_wchar_p), "value")),
        ]

    class IEnumSpellingError(IUnknown):
        _iid_ = GUID("{803E3BD4-2828-4410-8290-418D1D73C762}")
        _methods_ = [COMMETHOD([], HRESULT, "Next", (["out"], POINTER(POINTER(ISpellingError)), "value"))]

    class ISpellChecker(IUnknown):
        _iid_ = GUID("{B6FD0B71-E2BC-4653-8D05-F197E412770B}")
        _methods_ = [  # only the leading methods; the vtable order must match spellcheck.h
            COMMETHOD([], HRESULT, "get_LanguageTag", (["out"], POINTER(c_wchar_p), "value")),
            COMMETHOD([], HRESULT, "Check", (["in"], c_wchar_p, "text"),
                      (["out"], POINTER(POINTER(IEnumSpellingError)), "value")),
        ]

    class ISpellCheckerFactory(IUnknown):
        _iid_ = GUID("{8E018A9D-2415-4677-BF08-794EA61F94BB}")
        _methods_ = [
            COMMETHOD([], HRESULT, "get_SupportedLanguages", (["out"], POINTER(POINTER(IUnknown)), "value")),
            COMMETHOD([], HRESULT, "IsSupported", (["in"], c_wchar_p, "languageTag"),
                      (["out"], POINTER(c_int), "value")),
            COMMETHOD([], HRESULT, "CreateSpellChecker", (["in"], c_wchar_p, "languageTag"),
                      (["out"], POINTER(POINTER(ISpellChecker)), "value")),
        ]

    return ISpellCheckerFactory, GUID("{7AB36653-1796-484B-BDFA-E74F1DB7C1DC}")


_INTERFACES = None
_INTERFACES_LOCK = threading.Lock()


def _match_existing_com_mode():
    """comtypes initializes COM for the thread that first imports it, as single-threaded unless
    sys.coinit_flags says otherwise, and that import fails outright if the thread already uses
    the multithreaded mode. The recorder's audio library (soundcard) does exactly that, so
    match whatever mode this thread already has before comtypes is imported (as mic_mute does)."""
    if "comtypes" in sys.modules or hasattr(sys, "coinit_flags"):
        return
    import ctypes
    kind, qualifier = ctypes.c_int(), ctypes.c_int()
    if ctypes.windll.ole32.CoGetApartmentType(ctypes.byref(kind), ctypes.byref(qualifier)) == 0 and kind.value == 1:
        sys.coinit_flags = 0  # COINIT_MULTITHREADED, the mode this thread is already in


def _checker():
    """This thread's spell checker, or None when there isn't one."""
    if hasattr(_local, "checker"):
        return _local.checker
    _local.checker = None
    if sys.platform != "win32":
        return None
    global _INTERFACES
    try:
        _match_existing_com_mode()
        import comtypes
        try:
            comtypes.CoInitializeEx()  # each thread needs COM; a thread already set up is fine
        except OSError:
            pass
        with _INTERFACES_LOCK:
            if _INTERFACES is None:
                _INTERFACES = _interfaces()
        factory_interface, clsid = _INTERFACES
        factory = comtypes.CoCreateInstance(clsid, interface=factory_interface,
                                            clsctx=comtypes.CLSCTX_INPROC_SERVER)
        if factory.IsSupported(_LANGUAGE):
            _local.checker = factory.CreateSpellChecker(_LANGUAGE)
    except Exception:
        _local.checker = None
    return _local.checker


_answers: dict[str, bool] = {}
_MAX_ANSWERS = 20000


def is_english_word(word: str) -> bool | None:
    """True for a correctly spelled English word, False for a misspelling, None if unknown."""
    if word in _answers:
        return _answers[word]
    checker = _checker()
    if checker is None:
        return None
    try:
        answer = not bool(checker.Check(word).Next())
    except Exception:
        return None
    if len(_answers) < _MAX_ANSWERS:
        _answers[word] = answer
    return answer
