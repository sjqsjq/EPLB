import json, sys, urllib.request
API="http://127.0.0.1:30000/v1/completions"; OUT=sys.argv[1]; MODEL=sys.argv[2]
DS="/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L256_O1_realprover_n16384.jsonl"
N=int(sys.argv[3]) if len(sys.argv)>3 else 20
prompts=[]
with open(DS) as f:
    for i,line in enumerate(f):
        if i>=N: break
        prompts.append(json.loads(line)["prompt"][:1500])
res=[]
for p in prompts:
    body=json.dumps({"model":MODEL,"prompt":p,"max_tokens":1,"temperature":0.0,
                     "echo":True,"logprobs":1}).encode()
    req=urllib.request.Request(API, body, {"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        lp=json.loads(r.read())["choices"][0]["logprobs"]
    res.append(lp["token_logprobs"])
json.dump(res, open(OUT,"w"))
tot=sum(x for s in res for x in s if x is not None); n=sum(1 for s in res for x in s if x is not None)
print("saved",OUT,"tokens=",n,"sum_logprob=%.4f mean=%.6f"%(tot,tot/n))
