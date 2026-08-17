#!/usr/bin/env python3
"""Create representative B3 failure visualizations from completed FULL evidence."""
from __future__ import annotations
import ast, csv, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"third_party/nanosam"))
from nanosam.utils.predictor import Predictor
OUT=ROOT/"results/algorithm_b3_nanosam_full"; ANN=ROOT/"data/coco/annotations/instances_val2017.json"; IMAGES=ROOT/"data/coco/val2017"

def load_rows():
    rows={}
    with (OUT/"per_instance_primary.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            r["annotation_id"]=int(r["annotation_id"]); r["image_id"]=int(r["image_id"]); r["oracle_index"]=int(r["oracle_index"]); r["bbox"]=ast.literal_eval(r["bbox"]); r["predicted_iou"]=ast.literal_eval(r["predicted_iou"]); r["actual_iou"]=ast.literal_eval(r["actual_iou"]); r["slot0_iou"]=float(r["slot0_iou"]); r["oracle_iou"]=float(r["oracle_iou"]); r["oracle_headroom"]=float(r["oracle_headroom"]); rows[r["annotation_id"]]=r
    return rows

def overlay(im,mask,color):
    a=im.convert("RGBA"); rgba=np.zeros((mask.shape[0],mask.shape[1],4),np.uint8); rgba[mask]=(*color,115); return Image.alpha_composite(a,Image.fromarray(rgba,"RGBA")).convert("RGB")
def panel(im,text):
    c=Image.new("RGB",(im.width,im.height+52),"white"); c.paste(im,(0,52)); ImageDraw.Draw(c).multiline_text((5,5),text,fill="black",font=ImageFont.load_default(),spacing=2); return c

def main():
    rows=load_rows(); failures=json.loads((OUT/"failure_cases.json").read_text()); groups=[("lowest",failures["lowest_iou_20"]),("small",failures["small_lowest_iou_20"]),("overconfident",failures["high_confidence_low_actual_20"]),("headroom",failures["largest_oracle_headroom_20"])]
    chosen=[]; seen_ann=set(); seen_img=set()
    for name,items in groups:
        n=0
        for item in items:
            aid=int(item["annotation_id"]); iid=int(item["image_id"])
            if aid in seen_ann or iid in seen_img: continue
            chosen.append((name,rows[aid])); seen_ann.add(aid); seen_img.add(iid); n+=1
            if n==5: break
    coco=COCO(str(ANN)); imgs={int(x["id"]):x for x in coco.dataset["images"]}; pred=Predictor(str(ROOT/"third_party/nanosam/data/resnet18_image_encoder.engine"),str(ROOT/"third_party/nanosam/data/mobile_sam_mask_decoder.engine")); vis=OUT/"visualizations"; vis.mkdir(exist_ok=True)
    evidence=[]
    for rank,(group,r) in enumerate(chosen[:20],1):
        im=Image.open(IMAGES/imgs[r["image_id"]]["file_name"]).convert("RGB"); pred.set_image(im); torch.cuda.synchronize(); x,y,w,h=r["bbox"]; hi,scores,_=pred.predict(points=np.asarray([[x,y],[x+w,y+h]],np.float32),point_labels=np.asarray([2,3],np.int64)); torch.cuda.synchronize(); masks=(hi[0]>0).cpu().numpy(); gt=coco.annToMask(coco.loadAnns([r["annotation_id"]])[0])>0
        orig=im.copy(); ImageDraw.Draw(orig).rectangle((x,y,x+w,y+h),outline=(255,30,30),width=3); oi=r["oracle_index"]
        ps=[panel(orig,f"original + GT box\n{r['category_name']} ann={r['annotation_id']}"),panel(overlay(im,gt,(20,190,60)),"GT mask"),panel(overlay(im,masks[0],(220,40,40)),f"fixed slot0 IoU={r['slot0_iou']:.4f}\npredicted={r['predicted_iou'][0]:.4f}"),panel(overlay(im,masks[oi],(225,165,20)),f"oracle slot={oi} IoU={r['oracle_iou']:.4f}\nheadroom={r['oracle_headroom']:.4f}")]
        resized=[]
        for p in ps:
            if p.width>330:p=p.resize((330,int(p.height*330/p.width)),Image.Resampling.BILINEAR)
            resized.append(p)
        canvas=Image.new("RGB",(sum(p.width for p in resized),max(p.height for p in resized)),"white"); xx=0
        for p in resized:canvas.paste(p,(xx,0));xx+=p.width
        dest=vis/f"{rank:02d}_{group}_ann_{r['annotation_id']}.jpg";canvas.save(dest,quality=90);evidence.append({"group":group,"annotation_id":r["annotation_id"],"image_id":r["image_id"],"category":r["category_name"],"size":r["size_bucket"],"slot0_iou":r["slot0_iou"],"oracle_headroom":r["oracle_headroom"],"path":str(dest.relative_to(ROOT))})
    (OUT/"visualizations.json").write_text(json.dumps({"count":len(evidence),"cases":evidence},indent=2)+"\n"); print(json.dumps({"visualizations":len(evidence)},indent=2))
if __name__=="__main__":main()
