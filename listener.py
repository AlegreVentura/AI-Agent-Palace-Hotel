# listener.py
from fastapi import FastAPI, WebSocket
import asyncio, time
from collections import defaultdict
from agent import run_agent  # wraps part_1_graph.invoke → Ui JSON

app = FastAPI()
buffers = defaultdict(lambda: {"tail": "", "last_call": 0})

def trim_tail(s, max_chars=2000):
    return s[-max_chars:]

@app.websocket("/ingest")
async def ingest(ws: WebSocket):
    await ws.accept()
    call_id = None
    while True:
        msg = await ws.receive_json()
        # expected: {"type":"transcript", "call_id":"abc", "speaker":"customer|agent",
        #            "text":"...", "final":true|false}
        call_id = msg["call_id"]
        if msg["speaker"] == "customer":
            buf = buffers[call_id]
            buf["tail"] = trim_tail((buf["tail"] + " " + msg["text"]).strip())
            if msg.get("final"):
                now = time.time()
                if now - buf["last_call"] > 0.6:      # debounce 600ms
                    ui = run_agent(call_transcription=buf["tail"],
                                   customer_info={}, agent_info={})
                    buf["last_call"] = now
                    # push to dashboard
                    await ws.send_json({"type":"ui.update", "call_id":call_id, "payload":ui})
