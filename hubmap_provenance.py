"""Public HuBMAP metadata: all kidney PAS datasets (with tissue block) and the link of the 12 challenge
donors to HuBMAP donor records (age, sex, race, height and weight)."""
import json
import os
import urllib.request

import pandas as pd

from config import KAGGLE_DIR, RES

SEARCH = "https://search.api.hubmapconsortium.org/v3/portal/search"


def search(body):
    req = urllib.request.Request(SEARCH, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))["hits"]["hits"]


def first(x):
    return x[0] if isinstance(x, list) and x else x


def datasets():
    hits = search({"size": 1000, "query": {"bool": {"must": [
        {"term": {"entity_type.keyword": "Dataset"}}, {"term": {"dataset_type.keyword": "Histology"}},
        {"match": {"origin_samples.organ": "LK RK"}}]}},
        "_source": ["hubmap_id", "uuid", "dataset_info", "metadata", "donor", "source_samples", "origin_samples",
                    "descendant_counts"]})
    rows = []
    for h in hits:
        s = h["_source"]
        md, mm = s.get("metadata", {}), s["donor"].get("mapped_metadata", {})
        src, org = (s.get("source_samples") or [{}])[0], (s.get("origin_samples") or [{}])[0]
        rows.append(dict(
            hubmap_id=s["hubmap_id"], uuid=s["uuid"], dataset_info=(s.get("dataset_info") or "").split(":")[-1].strip()[:60],
            stain=md.get("stain_name"),
            scanner=f"{md.get('acquisition_instrument_vendor', '')} {md.get('acquisition_instrument_model', '')}".strip(),
            res_um=md.get("resolution_x_value"), donor=s["donor"]["hubmap_id"], group=s["donor"].get("group_name"),
            age=first(mm.get("age_value")), sex=first(mm.get("sex")), race=first(mm.get("race")),
            bmi=first(mm.get("body_mass_index_value")), height=first(mm.get("height_value")),
            weight=first(mm.get("weight_value")), medical_history="; ".join(mm.get("medical_history") or []),
            organ=org.get("organ"), origin_description=org.get("description", "")[:120],
            section_sample=src.get("hubmap_id"), section_description=(src.get("description") or "")[:300].replace("\n", " | "),
            has_pyramid=bool((s.get("descendant_counts") or {}).get("dataset_type", {}).get("Histology [Image Pyramid]"))))
    ds = pd.DataFrame(rows).sort_values(["donor", "dataset_info"])
    sec = search({"size": 1000, "query": {"terms": {"hubmap_id.keyword": ds.section_sample.dropna().unique().tolist()}},
                  "_source": ["hubmap_id", "ancestors.hubmap_id", "ancestors.sample_category"]})
    block = {h["_source"]["hubmap_id"]: ";".join(a["hubmap_id"] for a in h["_source"].get("ancestors", [])
                                                if a.get("sample_category") == "block") for h in sec}
    ds["block"] = ds.section_sample.map(block)
    return ds


def link(ds):
    kd = pd.read_csv(os.path.join(KAGGLE_DIR, "HuBMAP-20-dataset_information.csv")).groupby("patient_number").first().reset_index()
    donors = ds.groupby("donor").first().reset_index()
    rows = []
    for _, k in kd.iterrows():
        cand = donors[(donors.age.astype(float) == k.age) & (donors.sex == k.sex) & (donors.race == k.race)]
        if pd.notna(k.weight_kilograms):
            cand = cand[((cand.weight.astype(float) - k.weight_kilograms).abs() < 0.6) &
                        ((cand.height.astype(float) - k.height_centimeters).abs() < 1)]
        sel = ds[ds.donor.isin(cand.donor)]
        rows.append(dict(patient_number=k.patient_number, age=k.age, sex=k.sex, race=k.race, bmi=k["bmi_kg/m^2"],
                         n_candidates=len(cand), hubmap_donor=";".join(cand.donor),
                         medical_history=";".join(cand.medical_history), group=";".join(cand.group),
                         n_pas_datasets=len(sel), scanners=";".join(sorted(set(sel.scanner))),
                         resolutions=";".join(sorted(set(map(str, sel.res_um))))))
    return pd.DataFrame(rows)


def main():
    ds = datasets()
    ds.to_csv(os.path.join(RES, "hubmap_pas_datasets.csv"), index=False)
    link(ds).to_csv(os.path.join(RES, "hubmap_donor_link.csv"), index=False)


if __name__ == "__main__":
    main()
