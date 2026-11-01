import asyncio, httpx, time
URL="http://localhost:30000/generate"
N=int(__import__('sys').argv[1]) if len(__import__('sys').argv)>1 else 256
# ~512-token prefill prompts (varied)
base=[
 "Explain the theory of relativity in detail, covering special and general relativity, the equivalence principle, gravitational time dilation, and experimental evidence.",
 "Write a comprehensive Python implementation of a thread-safe LRU cache with expiration, including tests and a discussion of time complexity.",
 "Describe the complete process of cellular respiration including glycolysis, the Krebs cycle, oxidative phosphorylation, and the role of ATP.",
 "Analyze the causes, major events, and long-term consequences of the Industrial Revolution across Europe and North America.",
]
def prompt(i):
    b=base[i%len(base)]
    return b*8  # ~512 tokens
async def one(c,i):
    r=await c.post(URL,json={"text":prompt(i),"sampling_params":{"max_new_tokens":1,"temperature":0}},timeout=180)
    return r.status_code
async def main():
    async with httpx.AsyncClient() as c:
        t0=time.time()
        codes=await asyncio.gather(*[one(c,i) for i in range(N)])
        dt=time.time()-t0
        ok=sum(1 for x in codes if x==200)
        print(f"N={N} ok={ok} time={dt:.1f}s prefill_throughput={ok/dt:.1f} req/s")
asyncio.run(main())
