import asyncio, aiohttp, time, statistics

URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
PROMPTS = ["你好", "解释量子计算", "写500字短文", "分析全球经济", "写快速排序"]

async def req(s, p):
    t = time.perf_counter()
    async with s.post(URL, headers=HEADERS, json={"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": p}], "max_tokens": 256}) as r:
        d = await r.json()
    return d["usage"]["completion_tokens"] / (time.perf_counter() - t)

async def bench(c):
    async with aiohttp.ClientSession() as s:
        tasks = [req(s, PROMPTS[i % 5]) for i in range(20)]
        sem = asyncio.Semaphore(c)
        res = await asyncio.gather(*[asyncio.ensure_task(asyncio.wait_for(t, None)) for t in tasks])
        print(f"Concurrency {c}: {sum(res)/len(res):.1f} tokens/s")

asyncio.run(bench(1))
asyncio.run(bench(4))
asyncio.run(bench(8))
asyncio.run(bench(16))