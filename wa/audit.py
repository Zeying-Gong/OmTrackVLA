"""Validate every episode's prompt and timebase; produce explicit failures."""
import argparse
import json
from pathlib import Path
from wa.data import TrackingData, audit

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--data-root")
    p.add_argument("--source-prefix")
    args=p.parse_args()
    result=dict(cache_sha256=audit(args.cache),status="PASS",splits={})
    for part in ("train","heldout"):
        dataset=TrackingData(args.cache,part,args.data_root,args.source_prefix)
        failures=[]
        for index in range(len(dataset.episodes)):
            try:
                dataset.episode_info(index)
            except (ValueError,KeyError,FileNotFoundError,OSError) as error:
                failures.append(dict(episode=index,error=str(error)))
        result["splits"][part]=dict(episodes=len(dataset.episodes),windows=len(dataset),failures=failures)
        if failures:
            result["status"]="BLOCKED_PROMPT_DATA"
    with open(args.output,"x") as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="splits"}))
    return 0 if result["status"]=="PASS" else 2

if __name__=="__main__":
    raise SystemExit(main())
