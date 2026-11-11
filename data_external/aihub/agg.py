import zipfile, json, glob, os, pandas as pd
S=os.path.dirname(os.path.abspath(__file__)); rows=[]
for z in sorted(glob.glob(os.path.join(S,"TL_*.zip"))):
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if not n.endswith(".json"): continue
            j=json.loads(zf.read(n).decode("utf-8")); a=j.get("file_attributes",{}); g=j.get("growth_indicators",{}) or {}
            r=dict(cat=os.path.basename(z)[3:-4],farm=a.get("farmId"),week=a.get("week"),plant=a.get("number"),truss=a.get("flowerCluster"),date=a.get("date"),n_shapes=len(j.get("shapes",[])))
            r.update({f"g_{k}":v for k,v in g.items()}); rows.append(r)
df=pd.DataFrame(rows); df.to_csv(os.path.join(S,"labels.csv"),index=False)
print("rows",len(df)); print(df.groupby("cat").agg(n=("farm","size"),farms=("farm","nunique"),plants=("plant","nunique"),dates=("date","nunique"),dmin=("date","min"),dmax=("date","max")).to_string())
print([c for c in df.columns if c.startswith("g_")])
with zipfile.ZipFile(os.path.join(S,"TS_Timeseries.zip")) as zf: print(zf.namelist()[:20], len(zf.namelist()))
