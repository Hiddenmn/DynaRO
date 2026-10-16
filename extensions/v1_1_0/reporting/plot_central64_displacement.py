from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation,generate_binary_structure
CROP=(slice(32,96),)*3
TRAJECTORY=Path(__file__).resolve().parent/'inputs_3d/dev_parent_simulator.npz'

def contact_followup(initial,truth,prob,trajectory=TRAJECTORY):
    """Fixed step-50 simulator cohorts, followed to step 75 under unchanged S3.

    Neighbours are evaluated in the full parent before selecting the interior.
    Both bars divide by the SAME simulator-defined baseline site count.
    Prediction bars are increments p50-p75, not thresholded or conditional
    on the model's (different) remaining inventory. No new model execution.
    """
    with np.load(trajectory,allow_pickle=False) as data:
        connected=data['connected_pore'].astype(bool)
        actions=data['action_index']
        saved=data['labels']
        assert np.array_equal(saved[[0,50,75]],truth[[0,2,3]])
        assert np.array_equal(saved[0],initial)
    assert np.all(actions[50:75]==2)  # the original source's S3 index
    assert not (connected & (initial<2)).any()
    inlet=np.zeros_like(connected);inlet[:,:,0]=connected[:,:,0]
    source=((truth[2]==2)|inlet)&connected
    contact=binary_dilation(source,structure=generate_binary_structure(3,1))&connected
    region=np.zeros_like(connected);region[CROP]=True
    remaining=region&(truth[2]>=3)
    p0=prob[2].astype(np.float64);p1=prob[3].astype(np.float64)
    assert np.all(p1<=p0+1e-7)
    rows=[]
    for label,group in (
        ('water_contact',remaining&connected&contact),
        ('no_water_contact',remaining&connected&~contact),
        ('not_inlet_connected',remaining&~connected)):
        n=int(group.sum());converted=int((group&(truth[3]==2)).sum())
        before=float(p0[group].sum());after=float(p1[group].sum())
        assert n>0 and 0<=converted<=n and -1e-6<=before-after<=n
        row={'cohort':label,'baseline_step':50,'followup_step':75,
             'simulator_baseline_oil_sites':n,'simulator_converted_sites':converted,
             'prediction_expected_oil_at_baseline':before,
             'prediction_expected_oil_at_followup':after,
             'prediction_expected_converted_sites':before-after,
             'simulator_mobilized_percent':100*converted/n,
             'prediction_mobilized_percent':100*(before-after)/n,
             'prediction_baseline_remaining_percent':100*before/n}
        for k in range(3,8):
            row[f'initial_class_{k}_sites']=int((group&(initial==k)).sum())
        rows.append(row)
    assert [r['simulator_baseline_oil_sites'] for r in rows]==[3581,26236,1973]
    assert sum(r['simulator_baseline_oil_sites'] for r in rows)==int(remaining.sum())
    assert sum(r['simulator_converted_sites'] for r in rows)==int(
        (remaining&(truth[3]==2)).sum())
    assert rows[-1]['simulator_converted_sites']==0
    assert abs(rows[-1]['prediction_expected_converted_sites'])<1e-6
    return rows
