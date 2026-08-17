#!/usr/bin/env python3
"""Stage B3: NanoSAM full COCO prompt-conditioned instance-mask evaluation."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, resource, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/nanosam"))
from nanosam.utils.predictor import Predictor

STD_SHA = "19a00a836c2f1d1f236ea34175d12c2afc670b9cb5136b30e6b39bf7ca78f015"
STD_MIOU = 0.718127976997991
BUCKETS = ("small", "medium", "large")

def path(v):
    p = Path(v); return p if p.is_absolute() else ROOT / p

def rel(p):
    try: return str(Path(p).relative_to(ROOT))
    except ValueError: return str(p)

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def dump(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=2, sort_keys=False)+"\n")

def bucket(area):
    # COCO standard area buckets: small < 32^2, medium < 96^2, large otherwise.
    return "small" if area < 32*32 else ("medium" if area < 96*96 else "large")

def finite(v):
    try: return math.isfinite(float(v))
    except (TypeError, ValueError): return False

def audit(ann_path, image_root, manifest_path, out):
    raw=json.loads(Path(ann_path).read_text()); coco=COCO(str(ann_path))
    images={int(x["id"]):x for x in raw["images"]}; cats={int(x["id"]):x["name"] for x in raw["categories"]}
    ex=defaultdict(list); legal=[]; clipped=0
    for a in raw["annotations"]:
        aid=int(a["id"]); iid=int(a["image_id"])
        if int(a.get("iscrowd",0))==1:
            ex["crowd"].append({"annotation_id":aid,"image_id":iid}); continue
        im=images.get(iid)
        b=a.get("bbox")
        if im is None:
            ex["other"].append({"annotation_id":aid,"reason":"missing image metadata"}); continue
        if not isinstance(b,list) or len(b)!=4 or not all(finite(x) for x in b) or float(b[2])<=0 or float(b[3])<=0:
            ex["invalid_bbox"].append({"annotation_id":aid,"bbox":b}); continue
        x,y,w,h=map(float,b); W,H=float(im["width"]),float(im["height"])
        x1,y1=max(0,min(x,W)),max(0,min(y,H)); x2,y2=max(0,min(x+w,W)),max(0,min(y+h,H))
        if x2<=x1 or y2<=y1:
            ex["invalid_bbox"].append({"annotation_id":aid,"reason":"no positive image intersection","bbox":b}); continue
        clipped += int([x1,y1,x2-x1,y2-y1] != [x,y,w,h])
        if not finite(a.get("area")) or float(a["area"])<=0:
            ex["other"].append({"annotation_id":aid,"reason":"invalid area"}); continue
        if not a.get("segmentation"):
            ex["invalid_mask"].append({"annotation_id":aid,"reason":"empty segmentation"}); continue
        try: m=coco.annToMask(a)
        except Exception as e:
            ex["invalid_mask"].append({"annotation_id":aid,"reason":"decode error","error":repr(e)}); continue
        if m.shape!=(int(im["height"]),int(im["width"])) or int(m.sum())==0:
            ex["invalid_mask"].append({"annotation_id":aid,"reason":"empty/shape mismatch"}); continue
        if int(a.get("category_id",-1)) not in cats:
            ex["other"].append({"annotation_id":aid,"reason":"invalid category"}); continue
        if not (image_root/im["file_name"]).is_file():
            ex["other"].append({"annotation_id":aid,"reason":"missing image file"}); continue
        legal.append({"image_id":iid,"annotation_id":aid,"category_id":int(a["category_id"])})
    legal.sort(key=lambda x:(x["image_id"],x["annotation_id"]))
    manifest={"name":"nanosam_coco_full_instances","dataset":"COCO 2017 val2017","protocol_version":"stage_b3_v1","evaluation_unit":"one non-crowd COCO object instance","instances":legal}
    dump(manifest_path,manifest)
    counts={k:len(v) for k,v in sorted(ex.items())}; total=len(raw["annotations"]); other=sum(counts.values())-counts.get("crowd",0)-counts.get("invalid_bbox",0)-counts.get("invalid_mask",0)
    audit={"dataset":"COCO 2017 val2017","annotations_path":rel(ann_path),"images_path":rel(image_root),"total_images":len(raw["images"]),"total_annotations":total,"total_categories":len(raw["categories"]),"excluded_crowd":counts.get("crowd",0),"excluded_invalid_bbox":counts.get("invalid_bbox",0),"excluded_invalid_mask":counts.get("invalid_mask",0),"other_excluded":other,"exclusion_counts":counts,"legal_instances":len(legal),"legal_unique_images":len({x["image_id"] for x in legal}),"bbox_clipped_legal_count":clipped,"accounting_check":len(legal)+sum(counts.values())==total,"protocol":{"crowd":"exclude because crowd region is not necessarily one unambiguous instance","bbox":"require finite positive xywh and positive image intersection; clip partial out-of-image coordinates","segmentation":"require decodable non-empty mask matching image shape","prompt":"clipped GT box corners as points with labels [2,3]","mask":"decoder high-resolution logits > 0","primary":"fixed slot 0","size":"COCO area: small < 32^2, medium < 96^2, large >= 96^2"},"exclusion_details":ex}
    dump(out/"protocol_audit.json",audit)
    meta={"path":rel(manifest_path),"sha256":sha(manifest_path),"size_bytes":Path(manifest_path).stat().st_size,"instances":len(legal),"unique_images":len({x["image_id"] for x in legal}),"frozen_for_future_comparisons":True}
    dump(out/"full_manifest_metadata.json",meta)
    return audit,meta

def materialize(manifest, coco, image_root):
    imgs={int(x["id"]):x for x in coco.dataset["images"]}; cats={int(x["id"]):x["name"] for x in coco.dataset["categories"]}; rows=[]
    for x in json.loads(Path(manifest).read_text())["instances"]:
        a=coco.loadAnns([int(x["annotation_id"])])[0]; im=imgs[int(a["image_id"])]
        bx,by,bw,bh=map(float,a["bbox"]); X,Y=float(im["width"]),float(im["height"])
        x1,y1=max(0,min(bx,X)),max(0,min(by,Y)); x2,y2=max(0,min(bx+bw,X)),max(0,min(by+bh,Y))
        rows.append({"annotation":a,"annotation_id":int(a["id"]),"image_id":int(a["image_id"]),"category_id":int(a["category_id"]),"category_name":cats[int(a["category_id"])],"area":float(a["area"]),"size_bucket":bucket(float(a["area"])),"bbox":[x1,y1,x2-x1,y2-y1],"image_path":image_root/im["file_name"]})
    return rows

def aggregate(rows, field):
    v=np.asarray([float(r[field]) for r in rows if r["status"]=="VALID"],dtype=np.float64)
    return {"count":int(v.size),"mean_iou":float(v.mean()) if v.size else None,"median_iou":float(np.median(v)) if v.size else None,"p10":float(np.percentile(v,10)) if v.size else None,"p25":float(np.percentile(v,25)) if v.size else None,"p50":float(np.percentile(v,50)) if v.size else None,"p75":float(np.percentile(v,75)) if v.size else None,"p90":float(np.percentile(v,90)) if v.size else None,"iou_ge_0_50_ratio":float(np.mean(v>=.5)) if v.size else None,"iou_ge_0_75_ratio":float(np.mean(v>=.75)) if v.size else None,"iou_ge_0_90_ratio":float(np.mean(v>=.9)) if v.size else None,"iou_lt_0_25_ratio":float(np.mean(v<.25)) if v.size else None}

def evaluate(manifest, ann_path, image_root, enc, dec, csv_path, out, label):
    coco=COCO(str(ann_path)); rows=materialize(manifest,coco,image_root); grouped=defaultdict(list); order=[]
    for r in rows:
        if r["image_id"] not in grouped: order.append(r["image_id"])
        grouped[r["image_id"]].append(r)
    print(f"loading {label}: instances={len(rows)} images={len(order)}",flush=True)
    pred=Predictor(str(enc),str(dec)); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0=time.perf_counter()
    rec=[]; fails=[]; enc_calls=dec_calls=0; enc_t=dec_t=metric_t=0.0; min_avail=None
    for ni,iid in enumerate(order,1):
        group=grouped[iid]
        try:
            im=Image.open(group[0]["image_path"]).convert("RGB")
            s=time.perf_counter(); pred.set_image(im); torch.cuda.synchronize(); enc_t+=time.perf_counter()-s; enc_calls+=1
        except Exception as e:
            for r in group: fails.append({"annotation_id":r["annotation_id"],"image_id":iid,"stage":"set_image","error":repr(e)})
            continue
        for r in group:
            a=r["annotation"]; x,y,w,h=r["bbox"]; row={"annotation_id":r["annotation_id"],"image_id":iid,"category_id":r["category_id"],"category_name":r["category_name"],"area":r["area"],"size_bucket":r["size_bucket"],"bbox":r["bbox"],"status":"FAILED"}
            try:
                s=time.perf_counter(); hi,scores,_=pred.predict(points=np.asarray([[x,y],[x+w,y+h]],np.float32),point_labels=np.asarray([2,3],np.int64)); torch.cuda.synchronize(); dec_t+=time.perf_counter()-s; dec_calls+=1
                if tuple(hi.shape[:2])!=(1,4) or tuple(scores.shape)!=(1,4) or not bool(torch.isfinite(hi).all()) or not bool(torch.isfinite(scores).all()): raise RuntimeError("decoder contract or non-finite output")
                s=time.perf_counter(); gt=coco.annToMask(a)>0; masks=(hi[0]>0).detach().cpu().numpy(); ps=scores[0].detach().float().cpu().numpy(); actual=np.asarray([np.count_nonzero(m&gt)/np.count_nonzero(m|gt) if np.count_nonzero(m|gt) else 0.0 for m in masks]); metric_t+=time.perf_counter()-s
                pi=int(np.argmax(ps)); oi=int(np.argmax(actual)); row.update({"status":"VALID","predicted_iou":[float(z) for z in ps],"actual_iou":[float(z) for z in actual],"slot0_iou":float(actual[0]),"predicted_argmax_index":pi,"predicted_argmax_iou":float(actual[pi]),"oracle_index":oi,"oracle_iou":float(actual[oi]),"oracle_headroom":float(actual[oi]-actual[0])})
            except Exception as e:
                row["error"]=repr(e); fails.append({"annotation_id":r["annotation_id"],"image_id":iid,"stage":"predict","error":repr(e)})
            rec.append(row)
        try:
            vals={z.split(":")[0]:int(z.split()[1]) for z in Path("/proc/meminfo").read_text().splitlines() if ":" in z}; a=vals.get("MemAvailable"); min_avail=a if min_avail is None else min(min_avail,a)
        except Exception: pass
        if ni%100==0 or ni==len(order): print(f"{label} progress images={ni}/{len(order)} valid={sum(x['status']=='VALID' for x in rec)} failed={len(fails)}",flush=True)
    wall=time.perf_counter()-t0
    fields=["annotation_id","image_id","category_id","category_name","area","size_bucket","bbox","status","error","predicted_iou","actual_iou","slot0_iou","predicted_argmax_index","predicted_argmax_iou","oracle_index","oracle_iou","oracle_headroom"]
    with open(csv_path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rec)
    runtime={"scope":"algorithm quality evaluation, not deployment FPS","wall_time_seconds":wall,"legal_instances":len(rows),"unique_images":len(order),"encoder_calls":enc_calls,"decoder_calls":dec_calls,"embedding_reuse_instances":max(0,enc_calls and dec_calls-enc_calls),"encoder_aggregate_seconds":enc_t,"decoder_aggregate_seconds":dec_t,"mask_metric_aggregate_seconds":metric_t,"images_per_second":len(order)/wall,"instances_per_second":len(rows)/wall,"valid_count":sum(x["status"]=="VALID" for x in rec),"failed_count":len(fails),"peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),"torch_cuda_peak_allocated_bytes":int(torch.cuda.max_memory_allocated()),"torch_cuda_peak_reserved_bytes":int(torch.cuda.max_memory_reserved()),"minimum_system_mem_available_bytes":int(min_avail*1024) if min_avail else None,"csv":{"path":rel(csv_path),"size_bytes":Path(csv_path).stat().st_size,"sha256":sha(csv_path)}}
    dump(out/"runtime_summary.json",runtime); dump(out/"failed_instances.json",fails); return rec,runtime,fails

def ranks(x):
    o=np.argsort(x,kind="mergesort"); out=np.empty(len(x)); sx=x[o]; i=0
    while i<len(x):
        j=i+1
        while j<len(x) and sx[j]==sx[i]: j+=1
        out[o[i:j]]=(i+j-1)/2.; i=j
    return out

def pred_analysis(rows):
    rows=[r for r in rows if r["status"]=="VALID"]
    p=np.asarray([z for r in rows for z in r["predicted_iou"]]); a=np.asarray([z for r in rows for z in r["actual_iou"]])
    def one(rs):
        pp=np.asarray([z for r in rs for z in r["predicted_iou"]]); aa=np.asarray([z for r in rs for z in r["actual_iou"]]);
        return {"candidate_masks":len(pp),"pearson":float(np.corrcoef(pp,aa)[0,1]),"spearman":float(np.corrcoef(ranks(pp),ranks(aa))[0,1]),"mae":float(np.mean(abs(pp-aa))),"rmse":float(np.sqrt(np.mean((pp-aa)**2))),"argmax_top1_accuracy":float(np.mean([r["predicted_argmax_index"]==r["oracle_index"] for r in rs])) if rs else None}
    o={"all":one(rows)}
    for b in BUCKETS:o[b]=one([r for r in rows if r["size_bucket"]==b])
    return o

def category_analysis(rows):
    g=defaultdict(list)
    for r in rows:
        if r["status"]=="VALID": g[(r["category_id"],r["category_name"])].append(r)
    vals=[]
    for (cid,name),rs in g.items():
        m=aggregate(rs,"slot0_iou"); om=aggregate(rs,"oracle_iou")
        vals.append({"category_id":cid,"category_name":name,"count":len(rs),"slot0_mIoU":m["mean_iou"],"median_iou":m["median_iou"],"iou_ge_0_5":m["iou_ge_0_50_ratio"],"iou_ge_0_75":m["iou_ge_0_75_ratio"],"oracle_mIoU":om["mean_iou"],"size_counts":dict(Counter(r["size_bucket"] for r in rs))})
    return {"all_categories":sorted(vals,key=lambda x:x["category_id"]),"best_10":sorted(vals,key=lambda x:(-x["slot0_mIoU"],-x["count"]))[:10],"worst_10":sorted(vals,key=lambda x:(x["slot0_mIoU"],-x["count"]))[:10],"caution":"Descriptive; count is shown and low-count categories do not support strong generalization."}

def analyze(rows, fails, out, standard):
    valid=[r for r in rows if r["status"]=="VALID"]; primary=aggregate(rows,"slot0_iou"); primary.update({"valid_count":len(valid),"failed_count":len(fails),"policy":"fixed slot 0"}); dump(out/"full_metrics.json",primary)
    size={}
    for b in BUCKETS:
        rs=[r for r in valid if r["size_bucket"]==b]; size[b]={"count":len(rs),"mIoU":aggregate(rs,"slot0_iou")["mean_iou"],"median":aggregate(rs,"slot0_iou")["median_iou"],"iou_ge_0_5":aggregate(rs,"slot0_iou")["iou_ge_0_50_ratio"],"iou_ge_0_75":aggregate(rs,"slot0_iou")["iou_ge_0_75_ratio"],"predicted_argmax_mIoU":aggregate(rs,"predicted_argmax_iou")["mean_iou"],"oracle_mIoU":aggregate(rs,"oracle_iou")["mean_iou"]}
    dump(out/"size_bucket_analysis.json",size)
    fixed=aggregate(valid,"slot0_iou"); pa=aggregate(valid,"predicted_argmax_iou"); oracle=aggregate(valid,"oracle_iou"); pd=Counter(r["predicted_argmax_index"] for r in valid); od=Counter(r["oracle_index"] for r in valid)
    selection={"fixed_slot0":fixed,"predicted_iou_argmax":pa,"oracle_best_of_4":oracle,"predicted_argmax_delta_vs_fixed":pa["mean_iou"]-fixed["mean_iou"],"oracle_headroom_vs_fixed":oracle["mean_iou"]-fixed["mean_iou"],"predicted_argmax_slot_distribution":{str(i):{"count":pd[i],"ratio":pd[i]/len(valid)} for i in range(4)},"oracle_slot_distribution":{str(i):{"count":od[i],"ratio":od[i]/len(valid)} for i in range(4)},"predicted_argmax_equals_oracle_ratio":float(np.mean([r["predicted_argmax_index"]==r["oracle_index"] for r in valid]))}
    dump(out/"mask_selection_full.json",selection); pi=pred_analysis(rows); dump(out/"predicted_iou_analysis.json",pi); cats=category_analysis(rows); dump(out/"category_analysis.json",cats)
    global_small=float(np.mean([r["size_bucket"]=="small" for r in valid])); csa=[]
    for c in cats["worst_10"]:
        rs=[r for r in valid if r["category_id"]==c["category_id"]]; small=sum(r["size_bucket"]=="small" for r in rs)/len(rs); csa.append({"category_id":c["category_id"],"category_name":c["category_name"],"count":len(rs),"category_mIoU":c["slot0_mIoU"],"small_ratio":small,"global_small_ratio":global_small,"by_size":{b:{"count":sum(r["size_bucket"]==b for r in rs),"mIoU":aggregate([r for r in rs if r["size_bucket"]==b],"slot0_iou")["mean_iou"]} for b in BUCKETS}})
    dump(out/"category_size_analysis.json",{"scope":"worst 10 categories","global_small_ratio":global_small,"analysis":csa,"caution":"Association only; does not prove size causes category result."})
    vals=np.asarray([r["slot0_iou"] for r in valid]); counts,_=np.histogram(vals,np.linspace(0,1,11)); dump(out/"iou_distribution.json",{"metric":"fixed slot0 GT mask IoU","bins":[{"lower":i/10,"upper":(i+1)/10,"count":int(counts[i])} for i in range(10)]})
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,4)); plt.hist(vals,bins=np.linspace(0,1,11),edgecolor="white"); plt.xlabel("Fixed slot 0 IoU"); plt.ylabel("Instances"); plt.title("NanoSAM Full COCO Prompt-Conditioned Mask IoU"); plt.tight_layout(); plt.savefig(out/"iou_distribution.png",dpi=140); plt.close()
    except Exception: pass
    def compact(r): return {k:r.get(k) for k in ("annotation_id","image_id","category_id","category_name","area","size_bucket","bbox","slot0_iou","predicted_iou","predicted_argmax_index","predicted_argmax_iou","oracle_index","oracle_iou","oracle_headroom")}
    low=sorted(valid,key=lambda r:(r["slot0_iou"],r["annotation_id"])); small=sorted([r for r in valid if r["size_bucket"]=="small"],key=lambda r:(r["slot0_iou"],r["annotation_id"])); head=sorted(valid,key=lambda r:(-r["oracle_headroom"],r["annotation_id"])); p75=float(np.percentile([r["predicted_iou"][0] for r in valid],75)); q25=float(np.percentile([r["slot0_iou"] for r in valid],25)); over=sorted([r for r in valid if r["predicted_iou"][0]>=p75 and r["slot0_iou"]<=q25],key=lambda r:-(r["predicted_iou"][0]-r["slot0_iou"]))
    fc={"lowest_iou_20":[compact(r) for r in low[:20]],"small_lowest_iou_20":[compact(r) for r in small[:20]],"high_confidence_low_actual_20":[compact(r) for r in over[:20]],"high_confidence_rule":{"predicted_slot0_p75":p75,"actual_slot0_p25":q25},"largest_oracle_headroom_20":[compact(r) for r in head[:20]],"failed_instances":fails}; dump(out/"failure_cases.json",fc)
    full={"mIoU":primary["mean_iou"],"median_iou":primary["median_iou"],"iou_ge_0_50_ratio":primary["iou_ge_0_50_ratio"],"iou_ge_0_75_ratio":primary["iou_ge_0_75_ratio"],**{f"{b}_mIoU":size[b]["mIoU"] for b in BUCKETS}}; delta={k:full[k]-standard[k] for k in full}; dump(out/"standard_vs_full.json",{"standard":standard,"full":full,"full_minus_standard":delta,"representativeness":"described by absolute deltas; no universal representativeness claim from one subset"})
    return primary,size,selection,pi,cats,fc

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--annotations",default="data/coco/annotations/instances_val2017.json"); ap.add_argument("--image-root",default="data/coco/val2017"); ap.add_argument("--standard-manifest",default="data/eval_manifests/standard_manifest.json"); ap.add_argument("--full-manifest",default="data/eval_manifests/nanosam_coco_full_instances.json"); ap.add_argument("--encoder",default="third_party/nanosam/data/resnet18_image_encoder.engine"); ap.add_argument("--decoder",default="third_party/nanosam/data/mobile_sam_mask_decoder.engine"); ap.add_argument("--output-dir",default="results/algorithm_b3_nanosam_full"); ap.add_argument("--max-instances",type=int); ap.add_argument("--audit-only",action="store_true"); args=ap.parse_args()
    ann=image_root=path(args.annotations); image_root=path(args.image_root); std=path(args.standard_manifest); full=path(args.full_manifest); enc=path(args.encoder); dec=path(args.decoder); out=path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    audit_data,meta=audit(ann,image_root,full,out)
    if args.audit_only: print(json.dumps({"legal":audit_data["legal_instances"],"excluded":audit_data["excluded_crowd"]})); return
    if sha(std)!=STD_SHA: raise RuntimeError("STANDARD manifest SHA changed")
    # Mandatory regression uses the existing B1 evaluator and frozen accepted manifest.
    gate=out/"standard_gate"; gate.mkdir(exist_ok=True)
    cmd=[sys.executable,str(ROOT/"scripts/eval_nanosam_mask_selection_b1.py"),"--manifest",rel(std),"--output-dir",rel(gate)]
    subprocess.run(cmd,check=True)
    b1=json.loads((gate/"policy_summary.json").read_text()); obs=b1["policies"]["fixed_slot_0"]["mIoU"]; gate_status="PASS" if abs(obs-STD_MIOU)<1e-9 and b1["policies"]["fixed_slot_0"]["failed"]==0 else "STANDARD_REGRESSION_FAIL"
    gate_json={"status":gate_status,"expected_mIoU":STD_MIOU,"observed_mIoU":obs,"absolute_delta":abs(obs-STD_MIOU),"manifest_sha256":sha(std),"fixed_slot0_metrics":b1["policies"]["fixed_slot_0"],"size":{b:b1["policies"]["fixed_slot_0"] for b in []}}
    dump(out/"standard_regression.json",gate_json)
    if gate_status!="PASS": print("STANDARD_REGRESSION_FAIL"); sys.exit(10)
    standard={"mIoU":obs,"median_iou":b1["policies"]["fixed_slot_0"]["median_iou"],"iou_ge_0_50_ratio":b1["policies"]["fixed_slot_0"]["iou_ge_0_50"],"iou_ge_0_75_ratio":b1["policies"]["fixed_slot_0"]["iou_ge_0_75"],"small_mIoU":b1["policies"]["fixed_slot_0"]["small_mIoU"],"medium_mIoU":b1["policies"]["fixed_slot_0"]["medium_mIoU"],"large_mIoU":b1["policies"]["fixed_slot_0"]["large_mIoU"]}
    manifest=json.loads(full.read_text()); instances=manifest["instances"][:args.max_instances] if args.max_instances else manifest["instances"]
    if args.max_instances:
        tmp=out/"smoke_manifest.json"; dump(tmp,{**manifest,"instances":instances}); eval_manifest=tmp
    else: eval_manifest=full
    rows, runtime, fails=evaluate(eval_manifest,ann,image_root,enc,dec,out/"per_instance_primary.csv",out,"FULL" if not args.max_instances else "SMOKE")
    if args.max_instances: print(json.dumps({"smoke_valid":sum(r["status"]=="VALID" for r in rows),"smoke_failed":len(fails)})); return
    primary,size,selection,pi,cats,fc=analyze(rows,fails,out,standard)
    dump(out/"artifact_integrity.json",{"encoder_engine":{"path":rel(enc),"sha256":sha(enc)},"decoder_engine":{"path":rel(dec),"sha256":sha(dec)},"encoder_onnx":{"path":rel(ROOT/"third_party/nanosam/data/resnet18_image_encoder.onnx"),"sha256":sha(ROOT/"third_party/nanosam/data/resnet18_image_encoder.onnx")},"decoder_onnx":{"path":rel(ROOT/"third_party/nanosam/data/mobile_sam_mask_decoder.onnx"),"sha256":sha(ROOT/"third_party/nanosam/data/mobile_sam_mask_decoder.onnx")},"nanoowl_engine":{"path":rel(ROOT/"third_party/nanoowl/data/owlvit_image_encoder_patch32.engine"),"sha256":sha(ROOT/"third_party/nanoowl/data/owlvit_image_encoder_patch32.engine")}})
    dump(out/"stage_b3_machine_summary.json",{"decision":"FULL_SEGMENTATION_BASELINE_ESTABLISHED" if not fails else "FULL_SEGMENTATION_BASELINE_WITH_FAILURES","protocol_audit":audit_data,"manifest":meta,"standard_regression":gate_json,"full_metrics":primary,"size":size,"mask_selection":selection,"predicted_iou":pi,"best_5":cats["best_10"][:5],"worst_5":cats["worst_10"][:5],"runtime":runtime,"failed":len(fails)})
    print(json.dumps({"decision":"FULL_SEGMENTATION_BASELINE_ESTABLISHED" if not fails else "FULL_SEGMENTATION_BASELINE_WITH_FAILURES","legal":audit_data["legal_instances"],"valid":primary["valid_count"],"failed":len(fails),"mIoU":primary["mean_iou"]},indent=2))
if __name__=="__main__": main()
