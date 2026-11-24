import json, sqlite3, sys, urllib.request, re
sys.path.insert(0, r"F:\work\Project\ai-accent")
from aiaccent.metrics import measure

DB = r"E:\astrbot\instances\a6f520fc-da2d-4bfa-ac8a-1071e456c3b4\core\data\data_v4.db"
c = sqlite3.connect("file:%s?mode=ro" % DB.replace("\\","/"), uri=True)
persona = c.execute("select system_prompt from personas where id=2").fetchone()[0]
hist = json.loads(c.execute("select content from conversations where user_id=? order by updated_at desc limit 1",
                            ("大肥鱼:FriendMessage:339204071",)).fetchone()[0])
def txt(m):
    cs = m.get("content") or []
    return "".join(x.get("text","") for x in cs if isinstance(x, dict))
msgs = []
for m in hist:
    t = txt(m).strip()
    if not t: continue
    t = re.sub(r"<system_reminder>.*?</system_reminder>", "", t, flags=re.S).strip()
    if not t: continue
    msgs.append({"role": m["role"], "content": t})
print("clean msgs:", len(msgs), "total chars", sum(len(m["content"]) for m in msgs))

cfg = json.load(open(r"E:\astrbot\instances\a6f520fc-da2d-4bfa-ac8a-1071e456c3b4\core\data\cmd_config.json", encoding="utf-8"))
key = [s for s in cfg["provider_sources"] if s.get("id")=="deepseek"][0]["key"][0]
base = "https://api.deepseek.com/v1/chat/completions"

def call(system, convo):
    body = {"model":"deepseek-flash","temperature":1.0,"max_tokens":400,
            "thinking":{"type":"disabled"},
            "messages":[{"role":"system","content":system}]+convo}
    req = urllib.request.Request(base, data=json.dumps(body).encode(), headers={"Content-Type":"application/json","Authorization":"Bearer "+key})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"]

prompts = [m["content"] for m in msgs if m["role"]=="user"][-8:]
hist20 = msgs[:-1][-20:]   # 除最后一条用户
for name, system, prefix in [("persona-only", persona, []), ("persona+history", persona, hist20)]:
    turns=[]
    for p in prompts:
        out = call(system, prefix+[{"role":"user","content":p}])
        out = re.sub(r"&&[^&]{1,20}&&", "", out)
        chunks=[x.strip() for x in out.split("\n") if x.strip()]
        if chunks: turns.append(chunks)
    r = measure(turns)
    print("%-18s n=%d share3=%.2f entropy=%.2f punct=%.3f p50=%s" % (
        name, len(turns), r["turn_share_3"], r["turn_chunks_entropy"], r["end_punct_rate"], r["chunk_len_p50"]))
