import asyncio, httpx, time, json
URL="http://localhost:30000/generate"
rows=[json.loads(l) for l in open('/data/minghua/sjq/OEPLBdata/datasets/multi_domain/multidomain_v2_out1.jsonl')]
prov=[r["prompt"] for r in rows if r['id'].startswith('seg1')][:64]   # prover (seg1)
book=[r["prompt"] for r in rows if r['id'].startswith('seg2')][:64]   # book (seg2)
prov4=[r["prompt"] for r in rows if r['id'].startswith('seg4')][:64] # prover (seg4, diff theorems)
prompts=prov+book+prov4+book   # 4 blocks: prover/book/prover/book, 3 switches, 256
async def one(c,p):
    r=await c.post(URL,json={"text":p,"sampling_params":{"max_new_tokens":1,"temperature":0}},timeout=180)
    return r.status_code
async def main():
    async with httpx.AsyncClient() as c:
        t0=time.time()
        codes=await asyncio.gather(*[one(c,p) for p in prompts])
        dt=time.time()-t0; ok=sum(1 for x in codes if x==200)
        print(f"4blk(prover/book/prover/book) N={len(prompts)} ok={ok} time={dt:.1f}s tput={ok/dt:.1f} req/s")
asyncio.run(main())
