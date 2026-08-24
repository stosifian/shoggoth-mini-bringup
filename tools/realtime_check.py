"""Standalone check for OpenAI Realtime API access — no robot, no audio.

Confirms three things the orchestrator needs before any hardware work:
  1. OPENAI_API_KEY is set and valid,
  2. the key has Realtime API access (entitlement),
  3. the requested model id is accepted and the GA protocol is reachable.

You only receive a `session.created` event if all three hold, so that event is
the definitive pass. This uses the GA endpoint (the old beta interface, which
the orchestrator still targets, was removed 2026-05-12):

    wss://api.openai.com/v1/realtime?model=<model>
    Authorization: Bearer $OPENAI_API_KEY      (no OpenAI-Beta header in GA)

Run from the shoggoth-mini dir, venv active, with your key exported:
    export OPENAI_API_KEY=sk-...
    python tools/realtime_check.py
    python tools/realtime_check.py --model gpt-realtime-2.1
    python tools/realtime_check.py --say "say hello in five words"   # optional text round-trip
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def run(model: str, say: str | None, check_session: bool, timeout: float) -> int:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("FAIL: OPENAI_API_KEY is not set. `export OPENAI_API_KEY=sk-...` first.")
        return 2

    url = f"wss://api.openai.com/v1/realtime?model={model}"
    headers = {"Authorization": f"Bearer {key}"}  # GA: no OpenAI-Beta header
    print(f"connecting: {url}")

    try:
        async with websockets.connect(url, additional_headers=headers) as ws:
            msg = await asyncio.wait_for(ws.recv(), timeout)
            data = json.loads(msg)
            t = data.get("type")
            if t == "error":
                err = data.get("error", {})
                code = err.get("code") or err.get("type", "?")
                print(f"CONNECTED, but no session — server error: {code}")
                print(f"  {err.get('message', '')}")
                if code == "insufficient_quota":
                    print("  => Auth + Realtime access are FINE (in-band error, not 401/403). "
                          "This is BILLING: add a payment method/credits at "
                          "platform.openai.com/account/billing, then re-run.")
                return 1
            if t != "session.created":
                print(f"UNEXPECTED first event: {t}")
                print(json.dumps(data, indent=2)[:800])
                return 1
            sess = data.get("session", {})
            print("PASS: session.created")
            print(f"  session id : {sess.get('id')}")
            print(f"  model      : {sess.get('model', model)}")
            print("  => key valid, Realtime access OK, model accepted, GA protocol reachable.")

            if check_session:
                ok = await _validate_session_update(ws, timeout)
                if not ok:
                    return 1
            if say:
                await _text_round_trip(ws, say, timeout)
            return 0

    except asyncio.TimeoutError:
        print(f"FAIL: connected but no message within {timeout}s.")
        return 1
    except Exception as e:  # websockets raises version-specific auth errors
        m = str(e)
        hint = ""
        if "401" in m or "invalid_api_key" in m or "Unauthorized" in m:
            hint = "  -> 401: bad/expired API key."
        elif "403" in m or "insufficient" in m or "must be verified" in m:
            hint = "  -> 403: key lacks Realtime access (org may need verification/enablement)."
        elif "404" in m or "model" in m.lower():
            hint = f"  -> model '{model}' may be invalid; try --model gpt-realtime."
        print(f"FAIL: {type(e).__name__}: {m}")
        if hint:
            print(hint)
        return 1


async def _validate_session_update(ws, timeout: float) -> bool:
    """Send the orchestrator's real migrated session.update and confirm the GA
    API accepts it (session.updated) rather than rejecting the schema (error)."""
    from shoggoth_mini.configs.orchestrator import OrchestratorConfig

    session = OrchestratorConfig(system_prompt="GA migration validation").get_session_config()
    print("\n--session: sending the orchestrator's migrated session.update (GA schema)...")
    await ws.send(json.dumps({"type": "session.update", "session": session}))
    try:
        while True:
            data = json.loads(await asyncio.wait_for(ws.recv(), timeout))
            t = data.get("type")
            if t == "session.updated":
                print("  PASS: session.updated — GA session schema accepted "
                      "(type/output_modalities/audio.input.turn_detection/tools all valid).")
                return True
            if t == "error":
                print("  FAIL: server rejected session.update — a field is wrong:")
                print("   ", json.dumps(data.get("error", data))[:500])
                return False
            # ignore unrelated events until session.updated/error
    except asyncio.TimeoutError:
        print(f"  FAIL: no session.updated/error within {timeout}s.")
        return False


async def _text_round_trip(ws, say: str, timeout: float) -> None:
    """Best-effort: ask for a short text reply and print it (schema may drift)."""
    print(f"\n--say: requesting text reply to: {say!r}")
    await ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": say}]},
    }))
    await ws.send(json.dumps({"type": "response.create"}))
    out = []
    try:
        while True:
            data = json.loads(await asyncio.wait_for(ws.recv(), timeout))
            t = data.get("type", "")
            if t.endswith("output_text.delta") or t.endswith("text.delta"):
                out.append(data.get("delta", ""))
            elif t == "response.done":
                break
            elif t == "error":
                print("  server error event:", json.dumps(data.get("error", data))[:300])
                return
    except asyncio.TimeoutError:
        print("  (timed out waiting for response.done)")
    reply = "".join(out).strip()
    print("  reply:", reply if reply else "(no text deltas — schema may have changed; "
          "session.created is still the authoritative pass)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-realtime",
                    help="Realtime model id (e.g. gpt-realtime, gpt-realtime-2.1)")
    ap.add_argument("--say", default=None,
                    help="optional: send this text and print the model's reply")
    ap.add_argument("--session", action="store_true",
                    help="also send the orchestrator's migrated session.update and "
                         "verify the GA API accepts it (validates the beta->GA schema)")
    ap.add_argument("--timeout", type=float, default=15.0)
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args.model, args.say, args.session, args.timeout)))


if __name__ == "__main__":
    main()
