import asyncio, httpx, time, json, sys
URL="http://localhost:30000/generate"
DS="/data/minghua/sjq/OEPLBdata/datasets/multi_domain/multidomain_v2_out1.jsonl"
rows=[json.loads(l) for l in open(DS)]
seg1=[r["prompt"] for r in rows if r["id"].startswith("seg1")][:128]   # prover
seg2=[r["prompt"] for r in rows if r["id"].startswith("seg2")][:128]   # book
prompts=seg1+seg2  # 256 mixed, crosses prover->book
async def one(c,p):
    r=await c.post(URL,json={"text":p,"sampling_params":{"max_new_tokens":1,"temperature":0}},timeout=180)
    return r.status_code
async def main():
    async with httpx.AsyncClient() as c:
        t0=time.time()
        codes=await asyncio.gather(*[one(c,p) for p in prompts])
        dt=time.time()-t0; ok=sum(1 for x in codes if x==200)
        print(f"mixed(prover+book) N={len(prompts)} ok={ok} time={dt:.1f}s throughput={ok/dt:.1f} req/s")
asyncio.run(main())
