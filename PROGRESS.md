# Progress
Updated: 2026-10-06 China
Historical record: archive/2026-10/PROGRESS_before_a4_20261006.md and archive/2026-10/a4_before_hybrid_reuse_20261006/.

## Active A4 hybrid revision
User explicitly asked why A3 experience was not reused and requested offline/online RL mixture to directly improveA2. Full4215 recollection is no longer the prerequisite. NewNASpackage a4_hybrid_replay_20261006 contains the revised plan and real-data reuse audit;no hybrid optimizer or formaljob yet.
All8430 A3 source manifests independently matchA2 identity/instruction/seed7/initialRGB/sourcehash.886589 archived transitions in1732NPZ (~13.63GiB) remain available. First/middle/last NPZ hash/schema/finite checked again;prior fullreplay audit verified all actuallatent/reward/done/IDs.
4760 successful attempts,2914 unique source episodes ever succeeded,including568 originalA2failures(STT137/DT217/AT214). These reflect multiple exploratory policies,not a singlefrozen SR andnot an oracle-combined achievable score.
Only8672 savedadapt RGB frames:8426 episodes withfirstframe only and4completeepisodes(AT3/DT28 eachtwopasses). Cached16x256 features beforetrainablereader supportlatent learning;they cannot reconstruct fullRGB history/querytaps forFlow training.
A4 firsthybrid proposal:freezepristineA2 encoder/MetaQuery/Flow/oldheads/controller andfixedA3projector;newreader/residualpolicy/value. ExactA2seed7+stepCUDAnoise z0+zero-initializeddelta,actorseesz0. A3a isactualtotalz.
Completeepisode MCterminalSR/value+advantageweightedactualaction regression+A2softresidualconstraint;thennew/oldreplaymix online. Success/failuredata retained. AvoidA3unconstrainedQmax;MCtargets describeoldmixedbehavior,not unbiasednewpolicyadvantage. Following/collision/Lost remain separatelyreported.
Full886589 residualsupport PASS with exactCUDA A2noise:rho0.5 supports0, rho2 supports15 actual28Dactions. Firsthybrid usesunboundedresidualmean zero-initialized+A2softconstraint;smallinitialexploration isnotglobalhardbound. CompleteMCindexPASS:all1732 NPZhashes and886589actualtracepairs;48.76MBindex/64.50MBderivedfiles,4760success3670failureepisodes retained. ChangingFlow requiresseparatecandidate/replayprotocol.
Multiplefrozencheckpoints screenedonfixed252fullhorizon(84/task),promotedfull4215. Persistentbest registry retainsoriginalA2 untilstrictfullSRimproves. Onlyactualfrozen4215 qualifiesashighestSR.

## Preserved old A4 alternative and gate
Olda4_sr_adaptation_20261006 fullcollection/ActionExpert-finetuning branch statusSUPERSEDED_FIRST_ROUTE_BY_A3_REPLAY_HYBRID. Its3979frozeninputs SHA b87cfadaa2350892ad8eb34f70fa5165bf316cac9c13ddae88f8910a3b44c6a6 andallshortdeveloper/teacherrescue evidencepreserved.
Earlier8A80048h submissionwasrejectedbyautomaticapprovalbeforeSSHexecution;nojob. APPROVAL_BLOCK.json remainsbinding. Currentmethodclarificationdoesnotapprovethatoldrequest. Do not retry via anotherinterface/resourcechange/delegate. Continueindependenthybriddevelopment;preparethenrequestapprovalforconcretereviseformalconfiguration,norepeatedstalequestion.

## Completed A3 and references
A3_61020/71943 SUCCEEDED;8430adapt,886589transitions,221392updates,final frozen4215. Independentstrictpairs and431802-step reward/policyauditPASS.
A3FINAL SHA c7269fa14ba53ec60f4f27fa80aa9b295f83464ca5e0124597c30e0c10fce435.
STT1061/1405 SR75.516014 TR81.245010 CR2.633452;DT583/1405 SR41.494662 TR59.191678 CR3.772242;AT565/1405 SR40.213523 TR70.669425 CR4.768683(percent).
Overall2209/4215 SR52.408066 TR70.368704 CR3.724792;SR-10.106762ppvsA2,-11.269276ppvsB.388improvements/814regressions;Lost1331vs802. Initialnoise mismatch,Qinflation,actorboundarysaturation andfinal-onlyselection documented;causalfailurecategoriesunannotated.
A2_60058/60857:2635/4215 SR62.514828%;STT1143,DT796,AT696. B60994/60996:2684/4215 SR63.677343%;STT1169,DT787,AT728,currenthighestmeasuredA2/B/A3.
Old60766STOPPED/60507OOMKilled137/60317SIGKILLunconfirmed preserved;donotresume. HistoricalA2action-only59752distinct,originalA60316fixed36mixedfrozen.
