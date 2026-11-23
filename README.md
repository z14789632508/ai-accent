# ai-accent

**AI 口音的体检表：不看词，看分布。**

![tests](https://github.com/z14789632508/ai-accent/actions/workflows/test.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.9%2B-blue)

给一段 AI 生成的对话（或者一个真人在线账号的全部回复），逐项量它离人类聊天有多远：
不是打一个总分，而是告诉你**哪一项偏了、偏多少、在人类的第几百分位**。

词表式"去 AI 味"解决的是症状。这个工具解决的是症状换了地方之后你还看不看得见。

## 30 秒

```bash
pip install -e .
aiaccent audit examples/sample.jsonl --register chat
```

真实输出（Python 3.12，`aiaccent 0.2.0`）：

```
corpus   examples\sample.jsonl
register chat        band=bootstrap_turns
baseline QQ 一对一私聊（本人 2757 条 + 9 位好友 379 条，共 3136 条文本消息）
turns 40  chunks 80

metric                       value     p10     p50     p90     pct     dev  verdict
--------------------------------------------------------------------------------------
chunk_len_p50                   12     7.2    12.0   15.84   45.0%     0.0  ok
chunk_len_cv                 1.037   1.086    1.81   2.534    7.8%   -1.07  LOW
end_punct_rate                0.05  0.0216   0.036  0.0504   88.8%     0.0  ok
turn_chunks_median             2.0    1.36     2.0     3.0   50.0%     0.0  ok
turn_share_3                 0.175  0.0822   0.137  0.1918   79.4%     0.0  ok
turn_chunks_entropy          1.825  1.8135  3.0225  4.2315   10.2%     0.0  ok
within_turn_cv_median        0.659  0.3072   0.512  0.7168   86.2%     0.0  ok

pct = 该值在人类分布中的百分位；dev = 偏离该侧带宽的倍数（跨指标不可比）
```

`examples/sample.jsonl` 是**合成语料**：只用来演示格式，不代表任何真实人群。换成你自己的导出（见「用」）才有意义。

## 这是怎么发现的

一个跑在 QQ 上的角色号（人设 9598 字）被实测了三件事：

| 观测 | 数值 |
|---|---|
| 人设长度 | 9,598 字 / 299 行 |
| 人设里的否定式规则 | 61 条（"别"字出现 59 次） |
| 回复段数分布 | `{3: 263, 2: 16, 4: 17, 1: 1, …}` ── **87.7% 恰好三段** |
| 段数熵 | 0.73 bit（人类同口径 1.15–2.56） |
| 段长 | 16 / 19 / 18 字（三段几乎等宽） |
| 词级八股命中 | **0.0%**（"总的来说""首先其次""希望对你有帮助"一个都没有） |

最后一行是关键：**它一个八股词都没说，但八股的味道全在**。
因为社区流行的那套做法（封词表）已经生效了 —— 生效的结果是八股从词汇层搬到了结构层。

本仓库把这叫 **suppression drift**，对应的假说叫 **八股守恒**：
> 封词不减少八股的总量，只改变它的形态。

## 为什么不是词表

1. **负向约束不可执行**。模型在"禁用词附近"采样的不是一个点，是一个连续空间。你封掉"首先"，它换"第一步"；封掉"总之"，它换"整体来看"。
2. **词是有限集，句式是生成空间**。你可以列 200 个词，列不完"把所有话都说成三段、每段都一样长、每段都收尾"。
3. **这是 Goodhart's Law**：指标是"不出现这些词"，目标是"像人"。指标被当成目标之后，被绕开是必然的。
4. **水床效应**：压词汇层，结构层鼓起来。实验 04 就是这个的一个对照实证。

所以本工具量的是**结构**：长度分布、条数分布、段内整齐度、句末标点率、句长熵。
这些都是"词表管不到"的量。

## 指标

| 指标 | 方向 | 意思 | 人（私聊） | 模型 |
|---|---|---|---|---|
| `chunk_len_p50` | ↑ 太长像模型 | 单条消息长度中位数 | 12 字 | 16 字 |
| `chunk_len_cv` | ↓ 太整齐像模型 | 全部消息长度的变异系数 | 1.81 | 0.36 |
| `end_punct_rate` | ↑ 太多句号像模型 | 以句末标点结尾的消息占比 | 3.6% | 8.2% |
| `turn_chunks_median` | ─ | 一次回复拆成几条 | 2 | 3 |
| `turn_share_3` | ↑ 越高越可疑 | "恰好三条"的轮次占比 | 13.7% | **44.5%** |
| `turn_chunks_entropy` | ↓ 越可预测越可疑 | "这次拆几条"的熵 | 3.02 bit | 1.96 |
| `within_turn_cv_median` | ↓ 越整齐越可疑 | 一轮内几条消息的长度 CV | 0.51 | 0.29 |
| `sentence_len_entropy` | (experimental) | 句长分布熵，需要 `--experimental` | 1.5–1.8 | — |

`sentence_len_entropy` 默认不显示：它对**语料口径**极度敏感（同一批人私聊 2.50、群聊 1.6），
不印出分箱时容易被误用，所以标成 experimental。

这两组值就是"同一批语料、同一份基线"的对照，复现见「实测」。

## 口径（这一节比指标重要）

**register 是一等公民。** `--register` 必填，没有默认值，因为拿群聊基线去量私聊是错的
（第一版就这么错过：基数是 77 个群、43 个人的群聊，被测样本却是一对一私聊）。

| register | 来源 | 说明 |
|---|---|---|
| `chat` | QQ 一对一私聊，3,136 条（本人 2,757 + 9 位好友 379） | 694 turns / 2,757 chunks |
| `chat_group` | QQ 群聊，77 个群、43 个说话人（每人 ≥300 条），526,350 条 | 68,709 turns / 523,962 chunks |

`chat` 只有 1 个说话人够 200 条，**估不出"人和人之间能差多少"**。所以 `chat` 的带在
bootstrap 抽样误差之外强制加 ±40% 的相对下限（`--band-floor 0.4`）：
宁可把带放宽，也不制造"±0.5 就算明显偏离"的假精度。基线文件里会写明 `band_method` 与 `band_floor`。

两个列不要混着读：

- `pct` = 这个值在人类分布里排第几。**跨指标可比**，也是唯一能直接念给人听的数字。
- `dev` = 偏离该侧带宽几倍。**只能同指标比**：上带和下带的宽度本来就不一样
  （例如 `chunk_len_cv` 上侧 1.086 / 下侧 0.716），拿 +5 和 −3 比是错的。

## 实测（同一批语料，两种版本）

留存量 490 条回复（2026-11 之前）：

```
turns 490  chunks 1481
chunk_len_p50               16±0.5     7.2    12.0   15.84   91.7%    1.04  HIGH
chunk_len_cv             0.362±0.02   1.086    1.81   2.534    0.0%    -2.0  LOW (明显)
end_punct_rate           0.035±0.01  0.0216   0.036  0.0504   44.5%     0.0  ok
turn_chunks_median         3.0±0.0    1.36     2.0     3.0   83.0%     0.0  ok
turn_share_3             0.89±0.03  0.0822   0.137  0.1918  100.0%   13.74  HIGH (明显)
turn_chunks_entropy      0.665±0.12  1.8135  3.0225  4.2315    1.2%   -1.95  LOW
within_turn_cv_median    0.198±0.02  0.3072   0.512  0.7168    1.5%   -1.53  LOW
```

全量 2,391 条回复（按"用户插话"切 turn）：

```
turns 2391  chunks 6930
chunk_len_p50               11±0.5     7.2    12.0   15.84   44.8%     0.0  ok
chunk_len_cv             1.763±0.65   1.086    1.81   2.534   47.7%     0.0  ok
end_punct_rate           0.082±0.01  0.0216   0.036  0.0504  100.0%    3.19  HIGH (明显)
turn_chunks_median           3±0.0    1.36     2.0     3.0   83.0%     0.0  ok
turn_share_3             0.445±0.02  0.0822   0.137  0.1918  100.0%    5.62  HIGH (明显)
turn_chunks_entropy      1.963±0.05  1.8135  3.0225  4.2315   12.3%     0.0  ok
within_turn_cv_median    0.288±0.01  0.3072   0.512  0.7168    7.1%   -1.09  LOW
```

两者差得很大，说明"八股"不是账号的固有属性，是**那段时间的写法**：
早期（490 条）几乎每轮恰好三段，全量摊薄到 44.5%，但"恰好三条"仍是人类（13.7%）的三倍多，
段长也仍然比人整齐。

## 三组对照：是词表在动，还是"多了一段提示"在动

只加一句"禁用这些词"，句末标点率和整齐度也变了 —— 但词表里没有一个字提到标点或分段。
所以补了 C 组：**等长、与风格无关**的叮嘱（"回答保持友善和耐心…"）。

300 条最易触发八股的 prompt（"请分析一下…""给出三条建议…""总结一下要点…"）上：

| 指标 | A 裸提示 | B 词表 | C 等长无关 |
|---|---|---|---|
| 词级命中 /千字 | 0.85 | **0.33** | 0.50 |
| 单段占比 | 5.7% | **43.3%** | 4.3% |
| 条数熵 | 3.245 | **2.522** | 3.194 |
| 句末标点率 | 0.783 | **0.938** | 0.825 |

**B 把词压掉 61%，同时把"每轮只回一段"从 5.7% 抬到 43.3%；等长无关的 C 几乎没动。**
所以整齐化不是"多了几十字提示"造成的，是词表本身造成的 —— 水床成立。

另一个诚实的结论：A 组只有 0.85 命中/千字，说明这个模型在裸提示下已经不太说这套词了。
**词表在治的那个病，现在已经不是主要症状。**

完整设置、判读、以及这张表不能证明什么：见 [`experiments/04_three_arm.md`](experiments/04_three_arm.md)。

## 用

```bash
pip install -e .

aiaccent --version
aiaccent baseline                          # 有哪些 register，各自多大、怎么算的带
aiaccent audit corpus.jsonl --register chat
aiaccent audit corpus.jsonl --register chat --ci          # 每项加 bootstrap 95% CI
aiaccent audit corpus.jsonl --register chat --prescribe   # 附带怎么改
aiaccent audit corpus.jsonl --register chat --json
aiaccent diff a.jsonl b.jsonl --register chat             # 两份语料对着看
aiaccent prescribe corpus.jsonl --register chat
aiaccent mine corpus.jsonl --register chat --k 20         # 从真人语料里挑最"像人"的 20 条当 few-shot
```

**输入格式**：JSONL，一行一个 turn，行内是这一轮拆成的几条消息：

```json
["在的"]
["刚吃完饭", "你呢"]
["哈哈哈哈这个我也看到了"]
```

也接受 `{"chunks": [...]}`、`{"text": "..."}`（按空行切 turn），以及纯文本（空行分 turn、单行成条）。

`--prescribe` 会给出采样参数和一张可以贴进 system prompt 的"形状卡"：

```
== shape card (贴进 system prompt) ==
回复形状：长度 2 到 30 字之间随便给，长短交错，允许一条只有两三个字；
经常只发一条就停，想补一句再发一条，不要攒成一大段；
句子可以断在半路，不用每段都收得整整齐齐；
不是每轮都要问回去，也不是每轮都要收尾总结。

== notes ==
  - [明显 13.7×带宽 · 人类百分位 100.0%] 恰好 3 段占 89%（人类 14%）：不是禁止三段，是让段数跟着内容走。
  - [明显 2.0×带宽 · 人类百分位 0.0%] 整体长度分布太窄（CV 0.36，人类 1.81）：有的轮次该长，有的该只有两个字。
```

## compare-provider：同一批 prompt，谁更像人

选模型时最想知道的事，以前只能靠感觉。

```bash
export DEEPSEEK_API_KEY=...
export MOONSHOT_API_KEY=...

aiaccent compare-provider prompts.jsonl \
  --provider deepseek:deepseek-chat \
  --provider moonshot:moonshot-v1-8k \
  --register chat --out-dir out/
```

同一份 prompt 分别打各家，各自存一份 turn 级 jsonl，再对同一份人类基线出两组指标，
末尾按"离人类区间多远"横向排序。供应商 base 用 `<VENDOR>_BASE_URL` 覆盖。

## 参数不都是有效的

`prescribe` 会给采样参数，但**不是每个供应商都支持**：

| 参数 | 通用 | DeepSeek |
|---|---|---|
| `temperature` | ✅ | ✅ |
| `max_tokens` | ✅ | ✅ |
| `top_p` | ✅ | ❌ |
| `presence_penalty` | ✅ | ❌ |
| `frequency_penalty` | ✅ | ❌ |

实测：DeepSeek 的 `presence_penalty` / `frequency_penalty` 传了会被忽略（不报错），
`top_p` 与 `temperature` 同时给可能冲突。所以 `--provider deepseek` 时，
处方里那些参数会被标成无效，只留形状卡 —— **这也是"参数调了半天没变化"的常见原因**。

## 三个坑

1. **采样单位必须是 turn**，不是单条消息。模型把三段塞进一条、还是分成三条发，
   决定了"条数分布"能不能被看见。按消息算，永远看不出三段结构。
2. **人类基线不能用自造样本**。你写 50 条"自以为像人"的样本，量出来的就是"你那 50 条的写法"。
   要真人的导出记录，且同一 register（私聊别配群聊）。
3. **`baselines/*.json` 只存统计量，不存文本**。这是故意的：语料是别人的聊天记录。

## 不做什么

- **不出总分**。一个数会被当成奖励函数去刷，然后就变成下一个八股。
- **不提供词表**。词表是这个仓库要解释的问题，不是答案。
- **不判断内容好坏**。它量形状，不量你说得对不对。

## 实验

| 文件 | 内容 |
|---|---|
| [`experiments/01_wordlist_effect.md`](experiments/01_wordlist_effect.md) | 可证伪的 A/B 设计（词表生效后结构会怎样） |
| [`experiments/02_baseline_qq_groups.md`](experiments/02_baseline_qq_groups.md) | 基线怎么来的、口径怎么选的、已知偏差 |
| [`experiments/03_ab_result.md`](experiments/03_ab_result.md) | A/B 实跑（各 300 条）：词级降 44%，结构变整齐 |
| [`experiments/04_three_arm.md`](experiments/04_three_arm.md) | 补 C 组对照：排除"提示长度"解释，水床成立 |

## 相关

- 这套东西最初是为了修一个 QQ 角色号的"鱼八股"：人设 9,598 字压到 2,848 字、否定句换成分布描述，
  三段率 89% → 14%，段数熵 0.67 → 1.34。
- 如果你也在写角色卡：**否定式规则是八股的最大来源**。写成"多数时候只回一条，偶尔两三条连发"，
  比写"最多三条"有效得多。

## License

MIT © 2026 moyu
