import asyncio, httpx, time, json, sys
URL="http://localhost:30000/generate"
CONC=int(sys.argv[1]) if len(sys.argv)>1 else 32
N=int(sys.argv[2]) if len(sys.argv)>2 else 1800
LABEL=sys.argv[3] if len(sys.argv)>3 else "freq6"
book=[json.loads(l)["prompt"] for l in open('/data/minghua/sjq/OEPLBdata/datasets/prefill_decode_correlation/book_4438tok_O10.jsonl')][:900]
prov=[json.loads(l)["prompt"] for l in open('/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_2048tok_out1.jsonl')][:900]
seg=N//6
prompts=[]
for i in range(6):
    src=book if i%2==0 else prov
    prompts+=src[:seg]
prompts=prompts[:N]
sem=asyncio.Semaphore(CONC)
async def one(c,p):
    async with sem:
        r=await c.post(URL,json={"text":p,"sampling_params":{"max_new_tokens":10,"temperature":0}},timeout=300)
        return r.status_code
async def main():
    async with httpx.AsyncClient() as c:
        t0=time.time()
        codes=await asyncio.gather(*[one(c,p) for p in prompts])
        dt=time.time()-t0; ok=sum(1 for x in codes if x==200)
        tps=ok/dt
        print(f"[{LABEL}] freq6 N={len(prompts)} conc={CONC} ok={ok} time={dt:.1f}s tps={tps:.2f}")
        json.dump({"label":LABEL,"N":len(prompts),"conc":CONC,"ok":ok,"time":dt,"tps":tps},
                  open(f"/workspace/EPLB/OEPLB/benchmarks/results/_freq6_{LABEL}.json","w"), indent=2)
asyncio.run(main())
