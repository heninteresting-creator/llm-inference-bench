import asyncio, aiohttp, time, statistics, json

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "/root/autodl-tmp/models/Qwen3.5-9B-W4A16"
HEADERS = {"Content-Type": "application/json", "Accept": "text/event-stream"}

with open("/root/autodl-tmp/bench/prompts.json", "r", encoding="utf-8") as f:
    PROMPTS = [p["p"] for p in json.load(f)]

async def req_stream(s, p, sem):
    """流式请求，精确测量 TTFT 和 TPOT"""
    async with sem:
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": p}],
            "max_tokens": 512,
            "stream": True,
        }
        
        t0 = time.perf_counter()
        first_token_time = None
        token_count = 0
        
        async with s.post(URL, headers=HEADERS, json=payload) as resp:
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk["choices"][0].get("delta", {}).get("content"):
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        token_count += 1
        
        t_end = time.perf_counter()
        
        ttft = first_token_time - t0 if first_token_time else t_end - t0
        tpot = (t_end - first_token_time) / (token_count - 1) if token_count > 1 else 0
        
        return t_end - t0, ttft, tpot, token_count

async def bench(c, n, warmup=4):
    sem = asyncio.Semaphore(c)
    async with aiohttp.ClientSession() as s:
        if warmup:
            await asyncio.gather(*[
                req_stream(s, PROMPTS[i % len(PROMPTS)], sem) for i in range(warmup)
            ])
        
        t_start = time.perf_counter()
        res = await asyncio.gather(*[
            req_stream(s, PROMPTS[i % len(PROMPTS)], sem) for i in range(n)
        ])
        wall_time = time.perf_counter() - t_start
    
    dts = sorted([x[0] for x in res])
    ttfts = [x[1] for x in res]
    tpots = [x[2] for x in res]
    toks = [x[3] for x in res]
    
    total_ct = sum(toks)
    wall_tput = total_ct / wall_time
    
    print(f"\n并发{c}: 请求数={n} 墙钟时间={wall_time:.1f}s")
    print(f"  墙钟吞吐={wall_tput:.1f}t/s  总生成token={total_ct}")
    print(f"  端到端 P50={statistics.median(dts):.2f}s P99={dts[int(len(dts)*0.99)]:.2f}s")
    print(f"  TTFT  P50={statistics.median(ttfts):.3f}s P99={sorted(ttfts)[int(len(ttfts)*0.99)]:.3f}s")
    print(f"  TPOT  P50={statistics.median(tpots):.3f}s P99={sorted(tpots)[int(len(tpots)*0.99)]:.3f}s")

async def main():
    for c in [1, 4, 8, 16, 32, 64]:
        n = 50 if c <= 8 else max(200, c * 5)
        await bench(c, n)

if __name__ == "__main__":
    asyncio.run(main())