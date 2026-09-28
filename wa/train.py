"""DDP-capable offline imitation baseline; not a closed-loop evaluation."""
import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from wa.data import TrackingData, audit, sha, PROTOCOL
from wa.model import Policy

def flags(batch, mode, device):
    n = len(batch["xy"])
    if mode == "all":
        ids = torch.randint(0,3,(n,),device=device)
        return ids != 1, ids != 0
    return (torch.full((n,),mode != "point",device=device,dtype=torch.bool),
            torch.full((n,),mode != "image",device=device,dtype=torch.bool))

def transfer(batch, device):
    return {k:v.to(device) for k,v in batch.items()}

def evaluate(model, dataset, batch_size, device, rank, world):
    # Unpadded rank shards: each heldout row counted exactly once.
    loader = DataLoader(Subset(dataset,range(rank,len(dataset),world)),batch_size=batch_size)
    model.eval()
    results = {}
    with torch.no_grad():
        for mode in ("image","point","mixed"):
            sums = torch.zeros(3,device=device,dtype=torch.float64)
            for batch in loader:
                batch = transfer(batch,device)
                image_valid,point_valid = flags(batch,mode,device)
                pred = model(batch["rgb"],batch["template"],batch["point"],batch["times"],image_valid,point_valid)["xy"]
                error = torch.linalg.vector_norm(pred-batch["xy"],dim=-1)
                sums += torch.stack((error.mean(1).sum(),error[:,-1].sum(),
                                     error.new_tensor(len(error)))).double()
            if world > 1:
                dist.all_reduce(sums)
            count = int(sums[2].item())
            results[mode] = dict(ADE_m=(sums[0]/count).item(),FDE_m=(sums[1]/count).item(),windows=count,
                                 SR=None,TR=None,collision_rate=None)
    return results

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--encoder-weights")
    p.add_argument("--diagnostic",action="store_true")
    p.add_argument("--limit",type=int)
    p.add_argument("--epochs",type=int,default=1)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--workers",type=int,default=0)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--lr",type=float,default=0.0003)
    p.add_argument("--world-weight",type=float,default=0.1)
    p.add_argument("--mode",choices=["image","point","mixed","all"],default="all")
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("--data-root")
    p.add_argument("--source-prefix")
    p.add_argument("--lane",choices=["managed","external-h100"],default="managed")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.world_weight < 0:
        p.error("epochs, batch size must be positive and world weight nonnegative")
    if not args.diagnostic and (args.limit is not None or not args.encoder_weights):
        p.error("Full runs require explicit pretrained weights and no subset limit")
    rank,world,local = (int(os.environ.get(k,d)) for k,d in
                        (("RANK","0"),("WORLD_SIZE","1"),("LOCAL_RANK","0")))
    if args.lane == "external-h100":
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if world != 8 or len(names)!=8 or not all("H100" in n for n in names):
            p.error("external-h100 requires torchrun with eight visible H100 GPUs")
    device = torch.device(f"cuda:{local}" if args.device=="cuda" else "cpu")
    if device.type=="cuda":
        torch.cuda.set_device(device)
    if world>1:
        dist.init_process_group("nccl" if device.type=="cuda" else "gloo")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    output = Path(args.output)
    # Every rank verifies the small audited cache; raw episode hashes are separate provenance.
    dataset_sha = audit(args.cache)
    train = TrackingData(args.cache,"train",args.data_root,args.source_prefix,args.limit)
    val = TrackingData(args.cache,"heldout",args.data_root,args.source_prefix,args.limit)
    sampler = DistributedSampler(train,num_replicas=world,rank=rank,seed=args.seed,drop_last=True) if world>1 else None
    loader = DataLoader(train,batch_size=args.batch_size,sampler=sampler,
                        shuffle=sampler is None,num_workers=args.workers,drop_last=True)
    if len(loader)==0:
        raise ValueError("Not enough rows for one global batch")
    model = Policy(args.encoder_weights).to(device).train()
    wrapped = DDP(model,device_ids=[local] if device.type=="cuda" else None,
                  find_unused_parameters=True) if world>1 else model
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=args.lr)
    if rank==0:
        output.mkdir(parents=True,exist_ok=False)
        config = vars(args)|dict(protocol_id=PROTOCOL,dataset_sha256=dataset_sha,
                 encoder_sha256=sha(args.encoder_weights) if args.encoder_weights else None,
                 effective_batch_size=args.batch_size*world,precision="fp32",
                 point_source="simulated_uwb",world_size=world)
        (output/"config.json").write_text(json.dumps(config,indent=2))
        root = Path(__file__).resolve().parents[1]
        env = dict(schema_version=1,commit=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),
                   dirty=subprocess.check_output(["git","-C",str(root),"status","--porcelain"],text=True).strip(),
                   torch=torch.__version__,cuda=torch.version.cuda,
                   gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        env["source_sha256"] = {str(f.relative_to(root)): sha(f) for f in sorted((root/"wa").rglob("*.py"))}
        (output/"environment.json").write_text(json.dumps(env,indent=2))
    if world>1:
        dist.barrier()
    start = time.monotonic()
    for epoch in range(args.epochs):
        wrapped.train()
        if sampler:
            sampler.set_epoch(epoch)
        losses = torch.zeros(3,device=device)
        for batch in loader:
            batch = transfer(batch,device)
            image_valid,point_valid = flags(batch,args.mode,device)
            optimizer.zero_grad(set_to_none=True)
            result = wrapped(batch["rgb"],batch["template"],batch["point"],batch["times"],image_valid,point_valid,
                             **dict(future=batch["future"],labels=batch["xy"],world_weight=args.world_weight))
            loss = result["loss"]
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses += torch.tensor([loss.item(),result["trajectory_loss"].item(),1.],device=device)
        if world>1:
            dist.all_reduce(losses)
        if rank==0:
            print(json.dumps(dict(epoch=epoch+1,loss=(losses[0]/losses[2]).item(),
                                  trajectory_loss=(losses[1]/losses[2]).item())),flush=True)
    metrics = evaluate(model,val,args.batch_size,device,rank,world)
    if rank==0:
        report = dict(status="DIAGNOSTIC_ONLY" if args.diagnostic else "OFFLINE_ONLY",
                      protocol_id=PROTOCOL,dataset_sha256=dataset_sha,seed=args.seed,
                      metrics=metrics,elapsed_s=time.monotonic()-start,
                      closed_loop=False,edge_latency_verified=False,point_source="simulated_uwb")
        (output/"metrics.json").write_text(json.dumps(report,indent=2))
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=config),
                   output/"checkpoint.pt")
        print(json.dumps(report,indent=2))
    if world>1:
        dist.destroy_process_group()

if __name__=="__main__":
    main()
