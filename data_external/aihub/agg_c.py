"""
AI-Hub 'Intelligent smart farm integrated data (tomato)' — label group c (fruit set) -> labels_c.csv

One row per label file in TL_c*.zip (training labels): greenhouse, plant, truss, survey date, the names of the
annotated shapes and the growth indicators stored in the file. src/external_aihub.py takes the first date with
numberOfFruitPerPlant > 0 as the fruit-set date of a truss (the count per truss is stored in that field;
numberOfFruitPerTruss repeats the truss number).

The AI-Hub archives are not redistributed. Download them from www.aihub.or.kr and put TL_c*.zip next to this script.
Usage: python data_external/aihub/agg_c.py
"""
import glob
import json
import os
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    rows = []
    for z in sorted(glob.glob(os.path.join(HERE, "TL_c*.zip"))):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if not n.endswith(".json"):
                    continue
                j = json.loads(zf.read(n).decode("utf-8"))
                a = j.get("file_attributes", {})
                g = j.get("growth_indicators", {}) or {}
                rows.append(dict(farm=a.get("farmId"), plant=a.get("number"), truss=a.get("flowerCluster"), date=a.get("date"),
                                 labels="|".join(sorted({s.get("label", "") for s in j.get("shapes", [])})),
                                 numberOfFruitPerPlant=g.get("numberOfFruitPerPlant"),
                                 numberOfFruitPerTruss=g.get("numberOfFruitPerTruss"),
                                 fruitingNode=g.get("fruitingNode"), numberOfTheFlower=g.get("numberOfTheFlower")))
    if not rows:
        raise SystemExit("no TL_c*.zip found next to this script")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "labels_c.csv"), index=False)
    print("rows", len(df), "| greenhouses", df["farm"].nunique(), "| plants", df.groupby("farm")["plant"].nunique().to_dict())


if __name__ == "__main__":
    main()
