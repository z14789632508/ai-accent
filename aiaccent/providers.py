# -*- coding: utf-8 -*-
"""compare-provider 的模型调用层：只用标准库，OpenAI 兼容 /chat/completions。

设计原则：
- 不内置任何 key，全部走环境变量：<VENDOR>_API_KEY / <VENDOR>_BASE_URL；
- 同一个 prompt 池、同一套采样参数，唯一变量是模型 —— 否则比出来的不是模型；
- 输出写成"一个 turn 一行"的 jsonl，直接喂给 audit。

用法（CLI 里）：
  aiaccent compare-provider prompts.jsonl --register chat \
      --provider deepseek:deepseek-flash --provider deepseek:deepseek-reasoner
"""

import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULTS = {
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

SPLIT = re.compile(r"\n\s*\n+")


def _endpoint(vendor):
    base = os.environ.get(vendor.upper() + "_BASE_URL") or DEFAULTS.get(vendor)
    if not base:
        sys.exit("unknown vendor %r: set %s_BASE_URL" % (vendor, vendor.upper()))
    return base.rstrip("/") + "/chat/completions"


def _key(vendor):
    k = os.environ.get(vendor.upper() + "_API_KEY")
    if not k:
        sys.exit("missing env %s_API_KEY" % vendor.upper())
    return k


def read_prompts(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    out.append(json.loads(line)["p"])
                    continue
                except (ValueError, KeyError):
                    pass
            out.append(line)
    return out


def _call(url, key, model, prompt, max_tokens, temperature, timeout=90):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + key,
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return (data["choices"][0]["message"].get("content") or "").strip()


def run(spec, prompts_path, out_dir, max_tokens=200, temperature=1.0, threads=6):
    vendor, _, model = spec.partition(":")
    if not model:
        sys.exit("provider must be vendor:model (got %r)" % spec)
    url, key = _endpoint(vendor), _key(vendor)
    prompts = read_prompts(prompts_path)
    if not prompts:
        sys.exit("empty prompts: %s" % prompts_path)

    def work(p):
        try:
            return _call(url, key, model, p, max_tokens, temperature)
        except Exception as e:      # 单条失败不拖垮整批
            print("  ! %s: %s" % (model, e), file=sys.stderr)
            return ""

    with ThreadPoolExecutor(max_workers=threads) as ex:
        texts = list(ex.map(work, prompts))

    os.makedirs(out_dir, exist_ok=True)
    name = "%s_%s" % (vendor, model.replace("/", "_"))
    path = os.path.join(out_dir, name + ".jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            if not t:
                continue
            chunks = [c.strip() for c in SPLIT.split(t) if c.strip()]
            f.write(json.dumps(chunks, ensure_ascii=False) + "\n")
    return name, path
