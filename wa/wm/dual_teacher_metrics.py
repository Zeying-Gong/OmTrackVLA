"""Full paired teacher metrics; no training release or student superiority claim."""
import math
from collections import Counter
from wa.wm.dual_teacher_selection import select_teacher, EXPERIMENT

def summarize_teachers(rows, manifest):
    expected={(task,e['key']) for task in ('stt','dt','at')
              for e in manifest['tasks'][task]['episodes']}
    keys=[(r['pair']['task'],r['pair']['key']) for r in rows]
    if len(set(keys))!=len(keys) or set(keys)!=expected:
        raise ValueError('missing, duplicate or unexpected paired outcomes')
    if not expected:raise ValueError('empty manifest')
    for row in rows:
        check=select_teacher(row['branches']['lightnav'],row['branches']['oracle'])
        for field in ('pair','selected_teacher','reason','results','demonstration_candidate'):
            if check[field]!=row[field]:raise ValueError('selection record mismatch')
    report={}
    ref=manifest['reference_steps']
    for task in ('stt','dt','at'):
        part=[r for r in rows if r['pair']['task']==task]
        if not part:raise ValueError('empty task')
        report[task]={}
        for teacher in ('lightnav','oracle'):
            results=[r['branches'][teacher]['result'] for r in part]
            denominator=0
            for row,result in zip(part,results):
                for field in ('following_step','total_step'):
                    value=result[field]
                    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                        raise ValueError('invalid step count')
                if result['following_step']>result['total_step']:raise ValueError('following exceeds total')
                reference=ref.get(row['pair']['key'],0)
                if isinstance(reference,bool) or not isinstance(reference,(int,float)) or not math.isfinite(reference) or reference<0:
                    raise ValueError('invalid reference count')
                denominator+=max(result['total_step'],reference)
            n=len(part)
            report[task][teacher]=dict(episodes=n,
                SR=100*sum(r['success'] for r in results)/n,
                CR=100*sum(r['collision'] for r in results)/n,
                TR=100*sum(r['following_step'] for r in results)/denominator if denominator else None,
                macro_TR=100*sum(r['following_rate'] for r in results)/n,
                TR_denominator=denominator,
                invalid_init_count=sum(not r['policy_init_valid'] for r in results),
                reference_missing=sum(r['pair']['key'] not in ref for r in part),
                fallback_episode_count=sum(bool(r['branches'][teacher].get('transport_fallback')) for r in part))
        report[task]['teacher_success_union_count']=sum(
            bool(r['results']['lightnav']['success'] or r['results']['oracle']['success']) for r in part)
        report[task]['selection_counts']=dict(Counter(r['selected_teacher'] or 'neither' for r in part))
    return dict(experiment=EXPERIMENT,paired_episodes=len(rows),metrics_percent=report,
        training_released=False,limits=[
            'Complete manifest coverage required; artifact integrity requires separate audit.',
            'Teacher success union is not a student score or guaranteed attainable bound.',
            'CR is target-person distance below 0.5m, not general obstacle contact.',
            'TR uses sum(following_step)/sum(max(total_step,reference_step)); missing references use total_step.',
            'Released LightNav fallback remains in its score; fallback demonstrations are excluded.',
            'Evaluation-set adaptation, not untouched-test generalization.'])
