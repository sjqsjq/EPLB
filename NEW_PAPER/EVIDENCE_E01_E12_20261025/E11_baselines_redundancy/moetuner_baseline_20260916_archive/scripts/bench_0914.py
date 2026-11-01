import asyncio, httpx, time, json, sys
URL="http://localhost:30000/generate"
DS=sys.argv[1] if len(sys.argv)>1 else "/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_256tok_out1.jsonl"
N=int(sys.argv[2]) if len(sys.argv)>2 else 256
LABEL=sys.argv[3] if len(sys.argv)>3 else "run"
prompts=[json.loads(l)["prompt"] for l in open(DS)][:N]
async def one(c,p):
    r=await c.post(URL,json={"text":p,"sampling_params":{"max_new_tokens":1,"temperature":0}},timeout=180)
    return r.status_code
async def main():
    async with httpx.AsyncClient() as c:
        t0=time.time()
        codes=await asyncio.gather(*[one(c,p) for p in prompts])
        dt=time.time()-t0; ok=sum(1 for x in codes if x==200)
        tp=ok/dt
        print(f"[{LABEL}] prover N={N} ok={ok} time={dt:.1f}s throughput={tp:.1f} req/s")
        json.dump({"label":LABEL,"N":N,"ok":ok,"time":dt,"tps":tp}, open(f"/workspace/EPLB/OEPLB/benchmarks/results/_0914_{LABEL}.json","w"), indent=2)
asyncio.run(main())
