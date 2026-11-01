import asyncio, httpx, time, json, sys
URL="http://localhost:30000/generate"
DS="/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_256tok_out1.jsonl"
N=int(sys.argv[1]) if len(sys.argv)>1 else 256
prompts=[json.loads(l)["prompt"] for l in open(DS)][:N]
async def one(c,p):
    r=await c.post(URL,json={"text":p,"sampling_params":{"max_new_tokens":1,"temperature":0}},timeout=180)
    return r.status_code
async def main():
    async with httpx.AsyncClient() as c:
        t0=time.time()
        codes=await asyncio.gather(*[one(c,p) for p in prompts])
        dt=time.time()-t0; ok=sum(1 for x in codes if x==200)
        print(f"prover N={N} ok={ok} time={dt:.1f}s throughput={ok/dt:.1f} req/s")
asyncio.run(main())
