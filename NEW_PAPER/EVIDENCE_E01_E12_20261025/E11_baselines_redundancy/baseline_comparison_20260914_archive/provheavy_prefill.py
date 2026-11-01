import asyncio, httpx, time, json
URL="http://localhost:30000/generate"
rows=[json.loads(l) for l in open('/data/minghua/sjq/OEPLBdata/datasets/multi_domain/multidomain_v2_out1.jsonl')]
prov=[r["prompt"] for r in rows if r['id'].startswith('seg1')][:192]   # 192 prover (pinned, high imb)
book=[r["prompt"] for r in rows if r['id'].startswith('seg2')][:64]    # 64 book (volatile)
prompts=prov+book   # 256, prover-heavy, 1 switch
async def one(c,p):
    r=await c.post(URL,json={"text":p,"sampling_params":{"max_new_tokens":1,"temperature":0}},timeout=180)
    return r.status_code
async def main():
    async with httpx.AsyncClient() as c:
        t0=time.time()
        codes=await asyncio.gather(*[one(c,p) for p in prompts])
        dt=time.time()-t0; ok=sum(1 for x in codes if x==200)
        print(f"provheavy(192prov+64book) N={len(prompts)} ok={ok} time={dt:.1f}s tput={ok/dt:.1f}")
asyncio.run(main())
