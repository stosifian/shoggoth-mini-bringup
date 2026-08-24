"""Mic-only runtime check for the GA-migrated Realtime event handling — no robot.

Exercises the exact OpenAI-facing path the orchestrator uses, driven by your
laptop mic, so you can confirm the beta->GA migration works end-to-end at
runtime WITHOUT the assembled robot:

  * audio capture -> input_audio_buffer.append (validates the audio input format)
  * server-VAD events (input_audio_buffer.speech_started/stopped/committed)
  * the RENAMED response events (response.output_text.delta/done)
  * function-call events (response.output_item.added,
    response.function_call_arguments.delta/done) — say "wave hello" or "nod yes"
    to make the model emit a perform_primitive tool call

It reuses the orchestrator's real `OrchestratorConfig.get_session_config()`, so
the session it sends is identical to production. Tool calls are printed and
acknowledged with a stub result (no motors move). This does NOT cover the robot
actuation (execute_behavior / closed-loop) — those stay hardware-gated.

Run from the shoggoth-mini dir, venv active, key exported:
    export OPENAI_API_KEY=sk-...
    python tools/realtime_mic_check.py
    python tools/realtime_mic_check.py --rate 24000   # if VAD never triggers at 16k

macOS will prompt for microphone permission for your terminal — grant it.
Speak, then pause; Ctrl+C to quit.
"""
import argparse
import asyncio
import base64
import json
import os
import sys
from pathlib import Path

import sounddevice as sd
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.configs.orchestrator import OrchestratorConfig  # noqa: E402


async def main_async(model: str, rate: int, block: int) -> int:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("FAIL: OPENAI_API_KEY not set."); return 2

    cfg = OrchestratorConfig(system_prompt=(
        "You are a playful tentacle robot. Keep replies to one short sentence. "
        "When the user asks you to wave, nod yes, shake no, or celebrate, call "
        "perform_primitive with the matching action."))
    session = cfg.get_session_config()

    url = f"wss://api.openai.com/v1/realtime?model={model}"
    headers = {"Authorization": f"Bearer {key}"}
    loop = asyncio.get_running_loop()
    audio_q: asyncio.Queue = asyncio.Queue()

    def mic_cb(indata, _frames, _t, status):
        if status:
            print("mic status:", status)
        if not loop.is_closed():
            try:
                loop.call_soon_threadsafe(audio_q.put_nowait, bytes(indata))
            except RuntimeError:
                pass

    print(f"connecting: {url}")
    async with websockets.connect(url, additional_headers=headers) as ws:
        # --- handshake + session config (same payload as the orchestrator) ---
        first = json.loads(await ws.recv())
        if first.get("type") != "session.created":
            print("unexpected first event:", first.get("type"), first); return 1
        print("session.created ok — sending migrated session.update...")
        await ws.send(json.dumps({"type": "session.update", "session": session}))

        calls: dict = {}

        async def sender():
            while True:
                chunk = await audio_q.get()
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode(),
                }))

        async def receiver():
            async for message in ws:
                data = json.loads(message)
                t = data.get("type", "")
                if t == "session.updated":
                    print("session.updated ok — SPEAK NOW (say 'hi', or 'can you wave?'). Ctrl+C to quit.\n")
                elif t == "input_audio_buffer.speech_started":
                    print("[VAD] speech started")
                elif t == "input_audio_buffer.speech_stopped":
                    print("[VAD] speech stopped")
                elif t == "input_audio_buffer.committed":
                    print("[VAD] committed -> requesting response")
                    await ws.send(json.dumps({"type": "response.create"}))
                elif t == "response.output_text.delta":   # RENAMED in GA
                    print(data.get("delta", ""), end="", flush=True)
                elif t == "response.output_text.done":
                    print("  <- [text done]")
                elif t == "response.output_item.added":
                    item = data.get("item", {})
                    if item.get("type") == "function_call":
                        calls[item["id"]] = {"name": item.get("name"), "args": "",
                                             "call_id": item.get("call_id")}
                elif t == "response.function_call_arguments.delta":
                    c = calls.get(data.get("item_id"))
                    if c:
                        c["args"] += data.get("delta", "")
                elif t == "response.function_call_arguments.done":
                    c = calls.pop(data.get("item_id"), None)
                    if c:
                        print(f"\n>> TOOL CALL: {c['name']}({c['args']})  [stub — no motors]")
                        # acknowledge like the orchestrator would, so the turn completes
                        await ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {"type": "function_call_output",
                                     "call_id": c["call_id"],
                                     "output": json.dumps({"status": "success"})}}))
                        await ws.send(json.dumps({"type": "response.create"}))
                elif t == "error":
                    print("\n[ERROR]", json.dumps(data.get("error", data))[:400])

        with sd.InputStream(samplerate=rate, blocksize=block, dtype="int16",
                            channels=1, callback=mic_cb):
            print(f"mic open: {rate} Hz, int16, mono, block {block}")
            await asyncio.gather(sender(), receiver())
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-realtime")
    ap.add_argument("--rate", type=int, default=16000,
                    help="mic sample rate (orchestrator default 16000; try 24000 if VAD is silent)")
    ap.add_argument("--block", type=int, default=2048, help="audio block size")
    args = ap.parse_args()
    try:
        sys.exit(asyncio.run(main_async(args.model, args.rate, args.block)))
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
