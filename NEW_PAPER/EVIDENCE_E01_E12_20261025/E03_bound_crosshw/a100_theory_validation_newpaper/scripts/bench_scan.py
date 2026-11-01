import asyncio, httpx, time, json, sys
URL="http://localhost:30000/generate"
DS=sys.argv[1]; N=int(sys.argv[2]); TAG=sys.argv[3]; RUN=int(sys.argv[4]); SAVE=int(sys.argv[5]); PFX=sys.argv[6] if len(sys.argv)>6 else "_scanA100_"
prompts=[json.loads(l)["prompt"] for l in open(DS)][:N]
async def one(c,p):
    r=await c.post(URL,json={"text":p,"sampling_params":{"max_new_tokens":1,"temperature":0}},timeout=180)
    return r.status_code
async def main():
    async with httpx.AsyncClient() as c:
        t0=time.time()
        codes=await asyncio.gather(*[one(c,p) for p in prompts])
        dt=time.time()-t0; ok=sum(1 for x in codes if x==200); err=N-ok
    print(f"[scan/{TAG}/r{RUN}] N={N} ok={ok} errors={err} time={dt:.2f}s tps={ok/dt:.2f}", flush=True)
    if SAVE:
        json.dump({"tag":TAG,"run":RUN,"N":N,"ok":ok,"errors":err,"total_time_s":dt,"tps":ok/dt},
                  open(f"/workspace/EPLB/OEPLB/benchmarks/results/{PFX}{TAG}_r{RUN}.json","w"),indent=2)
asyncio.run(main())
