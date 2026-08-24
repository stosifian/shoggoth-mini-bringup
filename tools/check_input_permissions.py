"""Preflight the macOS input permissions that `calibrate` and `trackpad` need.

Both tools drive motors from `pynput` global key listeners. A `pynput` Listener is a
listen-only **CGEventTap**, so it is governed by **Input Monitoring**
(`kTCCServiceListenEvent`) — NOT Accessibility (`kTCCServiceAccessibility`, which covers
*controlling* the UI). Granting only Accessibility changes nothing, and a denied tap is
still created successfully: no exception, no stderr, the tool simply hangs with dead
arrow keys.

Run this from the SAME terminal app you'll calibrate from, before touching the motors:

    python tools/check_input_permissions.py
    python tools/check_input_permissions.py --prompt   # let macOS register the right app

Pass -> both checks True; arrow keys will reach the calibration tool.
Fail -> follow the printed fix, then QUIT (Cmd+Q) and relaunch the terminal app; TCC
        state is read only at process launch, so a new window/tab inherits the denial.
"""
import argparse
import ctypes
import ctypes.util
import os
import subprocess
import sys

CF = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
AS = ctypes.cdll.LoadLibrary(
    "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
)
CG = ctypes.cdll.LoadLibrary(
    "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
)

ACCESSIBILITY_PANE = (
    "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
)
INPUT_MONITORING_PANE = (
    "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"
)


def is_trusted() -> bool:
    """Accessibility — controlling the UI. Not what pynput's Listener needs."""
    AS.AXIsProcessTrusted.restype = ctypes.c_bool
    return bool(AS.AXIsProcessTrusted())


def listen_event_access(request: bool = False):
    """Input Monitoring — observing input. This is the one that gates pynput."""
    try:
        fn = CG.CGRequestListenEventAccess if request else CG.CGPreflightListenEventAccess
        fn.restype = ctypes.c_bool
        return bool(fn())
    except AttributeError:
        return "unavailable (pre-10.15)"


def prompt_for_accessibility() -> bool:
    """AXIsProcessTrustedWithOptions(prompt=True) — makes macOS register the correct
    responsible app itself, instead of you picking a binary by hand."""
    key = ctypes.c_void_p.in_dll(AS, "kAXTrustedCheckOptionPrompt")
    true_val = ctypes.c_void_p.in_dll(CF, "kCFBooleanTrue")
    key_cb = ctypes.c_void_p.in_dll(CF, "kCFTypeDictionaryKeyCallBacks")
    val_cb = ctypes.c_void_p.in_dll(CF, "kCFTypeDictionaryValueCallBacks")

    CF.CFDictionaryCreate.restype = ctypes.c_void_p
    CF.CFDictionaryCreate.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_long,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    keys = (ctypes.c_void_p * 1)(key)
    vals = (ctypes.c_void_p * 1)(true_val)
    opts = CF.CFDictionaryCreate(
        None, keys, vals, 1, ctypes.byref(key_cb), ctypes.byref(val_cb)
    )

    AS.AXIsProcessTrustedWithOptions.restype = ctypes.c_bool
    AS.AXIsProcessTrustedWithOptions.argtypes = [ctypes.c_void_p]
    return bool(AS.AXIsProcessTrustedWithOptions(opts))


def ancestry():
    """Walk parent PIDs to the hosting .app — TCC grants attach to *that*, not python."""
    chain, pid = [], os.getpid()
    for _ in range(12):
        try:
            out = subprocess.run(
                ["ps", "-o", "ppid=,comm=", "-p", str(pid)],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            if not out:
                break
            ppid, comm = out.split(None, 1)
            chain.append((pid, comm))
            pid = int(ppid)
            if pid <= 1:
                break
        except Exception:
            break
    return chain


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prompt", action="store_true",
                    help="trigger the macOS trust dialogs instead of only reporting")
    args = ap.parse_args()

    if sys.platform != "darwin":
        print("Not macOS — these permissions don't apply.")
        return 0

    accessibility = is_trusted()
    input_monitoring = listen_event_access()

    print("=" * 72)
    print(f"  Accessibility      AXIsProcessTrusted():           {accessibility}")
    print(f"  Input Monitoring   CGPreflightListenEventAccess(): {input_monitoring}")
    print("=" * 72)
    print("\n  pynput's Listener needs INPUT MONITORING. Accessibility alone is not enough.")

    print(f"\n  host app (needs the grant): "
          f"{os.environ.get('__CFBundleIdentifier', 'unknown')}")
    print(f"  TERM_PROGRAM:               {os.environ.get('TERM_PROGRAM', 'unset')}")
    print("\n  process ancestry:")
    for pid, comm in ancestry():
        print(f"    {pid:>7}  {comm}")

    if args.prompt:
        print("\nRequesting Input Monitoring (approve the dialog)...")
        print(f"  -> {listen_event_access(request=True)}")
        print("Requesting Accessibility (approve the dialog)...")
        print(f"  -> {prompt_for_accessibility()}")
        print("\nNow QUIT the terminal app (Cmd+Q) and relaunch it.")

    if input_monitoring is True:
        print("\nOK: key listeners will work. Safe to run `calibrate` / `trackpad`.")
        return 0

    print(
        "\nFAIL: Input Monitoring is not granted — `calibrate` will hang silently.\n"
        f'  1. open "{INPUT_MONITORING_PANE}"\n'
        "  2. click + and add your terminal app, then enable it\n"
        "  3. QUIT it with Cmd+Q and relaunch (TCC is read at process launch)\n"
        "  4. re-run this check\n"
        f"  (Accessibility pane, if ever needed: {ACCESSIBILITY_PANE})"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
