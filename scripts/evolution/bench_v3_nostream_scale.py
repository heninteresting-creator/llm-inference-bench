# v3 非流式脚本：修正吞吐口径（墙钟）+ 跨越 --max-num-seqs 边界的并发扫描（16→192）
# 产出：results/s12-overload-cap64-0624.txt、results/s12-overload-default-0624.txt
# 注意：非流式（无 TTFT/TPOT）；读取 /root/prompts.json
import asyncio, aiohttp, time, statistics, json
from datetime import datetime

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
HEADERS = {"Content-Type": "application/json"}

with open("/root/prompts.json", "r", encoding="utf-8") as f:
    PROMPTS = [p["p"] for p in json.load(f)]

async def req(s, p, sem):
    """发送单个请求，记录耗时与 token 数"""
    async with sem:
        t0 = time.perf_counter()
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": p}],
            "max_tokens": 512,
        }
        async with s.post(URL, headers=HEADERS, json=payload) as resp:
            data = await resp.json()
        dt = time.perf_counter() - t0
        
        ct = data.get("usage", {}).get("completion_tokens", 0)
        pt = data.get("usage", {}).get("prompt_tokens", 0)
        return dt, ct, pt

async def bench(c, n, warmup=4):
    sem = asyncio.Semaphore(c)
    
    async with aiohttp.ClientSession() as s:
        # 预热，排除冷启动
        if warmup:
            await asyncio.gather(*[
                req(s, PROMPTS[i % len(PROMPTS)], sem) for i in range(warmup)
            ])
        
        # 正式压测：记录墙钟时间
        t_start = time.perf_counter()
        res = await asyncio.gather(*[
            req(s, PROMPTS[i % len(PROMPTS)], sem) for i in range(n)
        ])
        t_end = time.perf_counter()
    
    wall_time = t_end - t_start  # 实际经过时间（所有请求重叠执行）
    
    dts = sorted([x[0] for x in res])
    toks = [x[1] for x in res]
    pts = [x[2] for x in res]
    
    p50 = statistics.median(dts)
    p99 = dts[int(len(dts) * 0.99)] if len(dts) > 1 else dts[0]
    
    total_ct = sum(toks)
    total_pt = sum(pts)
    
    # 关键修正：墙钟吞吐 = 总生成 token / 实际经过时间
    # 旧脚本的 sum(toks)/sum(dts) 把并发重叠算成了串行，严重低估
    wall_tput = total_ct / wall_time if wall_time > 0 else 0
    
    # 单请求平均吞吐（供参考，无实际工程意义）
    avg_tput = total_ct / sum(dts) if sum(dts) > 0 else 0
    
    print(f"\n并发{c}: 请求数={n} 墙钟时间={wall_time:.1f}s")
    print(f"  墙钟吞吐={wall_tput:.1f}t/s  单请求平均={avg_tput:.1f}t/s  总生成token={total_ct}")
    print(f"  端到端 P50={p50:.2f}s P99={p99:.2f}s")
    
    return {
        "concurrency": c, "n": n, "wall_time": wall_time,
        "wall_tput": wall_tput, "avg_tput": avg_tput,
        "total_ct": total_ct, "total_pt": total_pt,
        "p50": p50, "p99": p99,
    }

async def main():
    # 关键改动：并发必须跨过 64，才能压出 max_num_seqs 的边界
    # 当服务端 --max-num-seqs=64 时：
    #   - 客户端并发 <=64：服务端无排队，延迟稳定
    #   - 客户端并发 >64：服务端内部排队，延迟暴涨，吞吐下降
    # 如果服务端是默认 256，则 128/192 仍流畅，这就是对比价值
    levels = [16, 32, 64, 96, 128, 192]
    
    all_res = []
    for c in levels:
        # 关键改动：高并发需要更多请求才能稳定，至少持续 20-30 秒
        n = max(200, c * 5)
        r = await bench(c, n)
        all_res.append(r)
    
    # 汇总表格
    print("\n" + "="*80)
    print(f"{'并发':>6} {'请求数':>8} {'墙钟吞吐(t/s)':>14} {'P50延迟':>10} {'P99延迟':>10}")
    print("-"*80)
    for r in all_res:
        print(f"{r['concurrency']:>6} {r['n']:>8} {r['wall_tput']:>14.1f} {r['p50']:>10.2f} {r['p99']:>10.2f}")
    
    # 保存结果，方便对比不同 max_num_seqs 配置
    fname = f"bench_{datetime.now().strftime('%m%d_%H%M')}.json"
    with open(fname, "w") as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {fname}")

if __name__ == "__main__":
    asyncio.run(main())