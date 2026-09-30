"""Policy-only synthetic history loss; original labels and WM sequence retained."""
import math
import torch
def repeat_current_history(batch,probability,generator=None):
    if not math.isfinite(probability) or not 0<=probability<=1:
        raise ValueError('history probability outside[0,1]')
    if probability==0:return batch
    rgb=batch['rgb'];times=batch['times']
    if rgb.ndim!=5 or rgb.shape[1]!=4 or times.shape!=rgb.shape[:2]:
        raise ValueError('expected Bx4 policy history')
    if not torch.isfinite(times).all() or (times>0).any() or not torch.all(times[:,-1]==0):
        raise ValueError('noncausal or uncentered history timestamps')
    mask=torch.rand(rgb.shape[0],device=rgb.device,generator=generator)<probability
    result=dict(batch)
    result['rgb']=torch.where(mask[:,None,None,None,None],rgb[:,-1:].expand_as(rgb),rgb)
    result['times']=torch.where(mask[:,None],torch.zeros_like(times),times)
    # template, polar, targets, wm_rgb, commands, proprio and future are untouched.
    return result

def verify_preserved(original,augmented):
    for key in original:
        if key not in ('rgb','times'):
            if augmented[key] is not original[key]:raise AssertionError(key+' changed')
    return True
