import asyncio, aiohttp, time, statistics, json

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
HEADERS = {"Content-Type": "application/json"}

with open("prompts.json", "r", encoding="utf-8") as f:
    PROMPTS = [p["p"] for p in json.load(f)]

async def req(s, p, sem, max_tokens):
    async with sem:
        t0 = time.perf_counter()
        payload = {"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": max_tokens, "temperature": 0.7}
        d = await (await s.post(URL, headers=HEADERS, json=payload)).json()
        dt = time.perf_counter() - t0
        ct = d["usage"]["completion_tokens"]
        pt = d["usage"]["prompt_tokens"]
        return dt, ct, pt

async def bench(c, n, max_tokens):
    sem = asyncio.Semaphore(c)
    async with aiohttp.ClientSession() as s:
        res = await asyncio.gather(*[req(s, PROMPTS[i % len(PROMPTS)], sem, max_tokens) for i in range(n)])
    
    dts = sorted([x[0] for x in res])
    toks = [x[1] for x in res]
    ptoks = [x[2] for x in res]
    
    p50 = statistics.median(dts)
    p99 = dts[int(len(dts)*0.99)]
    avg = sum(dts)/len(dts)
    agg_tput = sum(toks)/sum(dts)
    avg_tput = statistics.mean([t/d for t,d in zip(toks, dts)])
    
    return {
        "并发": c,
        "请求数": n,
        "P50(s)": round(p50, 2),
        "P99(s)": round(p99, 2),
        "平均延迟(s)": round(avg, 2),
        "总吞吐(t/s)": round(agg_tput, 1),
        "单请求平均吞吐(t/s)": round(avg_tput, 1),
        "总PromptTokens": sum(ptoks),
        "总GenTokens": sum(toks),
    }

async def main():
    max_tokens = 512
    n = 30
    rows = []
    for c in [1, 2, 4, 8, 16]:
        print(f"正在测试并发 {c} ...")
        row = await bench(c, n, max_tokens)
        rows.append(row)
    
    print("\n### 压测结果\n")
    header = "| " + " | ".join(rows[0].keys()) + " |"
    sep = "|" + "|".join(["---"] * len(rows[0])) + "|"
    print(header)
    print(sep)
    for r in rows:
        print("| " + " | ".join(str(v) for v in r.values()) + " |")
    
    with open("benchmark_result_v2.md", "w") as f:
        f.write("### 长Prompt压测结果V2\n\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(v) for v in r.values()) + " |\n")
    print("\n结果已保存到 benchmark_result_v2.md")

if __name__ == "__main__":
    asyncio.run(main())