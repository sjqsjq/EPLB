import asyncio, httpx, time, json
URL="http://localhost:30000/generate"
rows=[json.loads(l) for l in open('/data/minghua/sjq/OEPLBdata/datasets/multi_domain/prefill_heavy_universal.jsonl')]
prompts=[]
for d in ['sharegpt','prover','book','code']:
    prompts+=[r["prompt"] for r in rows if r.get("id","").startswith(d)][:64]
async def one(c,p):
    r=await c.post(URL,json={"text":p,"sampling_params":{"max_new_tokens":1,"temperature":0}},timeout=180)
    return r.status_code
async def main():
    async with httpx.AsyncClient() as c:
        t0=time.time()
        codes=await asyncio.gather(*[one(c,p) for p in prompts])
        dt=time.time()-t0; ok=sum(1 for x in codes if x==200)
        print(f"universal4(sharegpt/prover/book/code 64each) N={len(prompts)} ok={ok} time={dt:.1f}s tput={ok/dt:.1f}")
asyncio.run(main())
