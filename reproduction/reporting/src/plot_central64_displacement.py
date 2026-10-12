"""Interior displacement, not the boundary of the original 128^3 input.

Scientific question: how much oil changes in a geometrically fixed interior,
where is that change, and does the existing prediction reproduce it?
Image plate + quantification: a initial; b/c final truth/prediction; d converted
oil in initial context; e temporal amount; f contact-stratified subsequent
mobilization. Same cached
DEV case as Supplementary Figure S2; this is the main-text interior view.
Python, 180 x 123 mm; original purple/blue phase mapping, green/pink statistics.
One case and one seed, no population uncertainty or inferential test.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='2'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
from surface_renderer import (
    PALETTE, LIGHT, mesh, geometry_buffer, make_camera, light_pixels, project)
from figure_layout import check_layout as require_matplotlib_panel_alignment
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import psutil
from scipy.ndimage import binary_dilation, generate_binary_structure

SOURCE=HERE.parent/'inputs_3d/dev_parent_prediction.npz'
TRAJECTORY=HERE.parent/'inputs_3d/dev_parent_simulator.npz'
STEM='figure12_central64_displacement_v2'
GREEN,PINK,INK='#428F7D','#C97C9D','#293745'
CROP=(slice(32,96),)*3
CAMERA={'azimuth':-125.,'elevation':24.}
WIDTH,HEIGHT=180,123

def volume(mask,water=None,*,changed=False,ghost=None,pixels=1400):
    # Keep coordinates on the original grid for lighting; zero only material
    # outside the explicitly declared display crop. No rescaling of geometry.
    retained=np.zeros((128,)*3,bool);retained[CROP]=mask[CROP]
    c=make_camera(pixels,az=CAMERA['azimuth'],el=CAMERA['elevation'])
    c['span']*=.5
    m=mesh(retained,6 if changed else 7)
    depth,normals,categories=geometry_buffer(m,c)
    lc=make_camera(700,direction=LIGHT);lc['span']*=.5
    shadow,_,_=geometry_buffer(m,lc)
    rgb,_=light_pixels(depth,normals,categories,retained,PALETTE,
        c['center'],c['right'],c['up'],c['view'],c['span'],LIGHT,
        shadow,lc['center'],lc['right'],lc['up'],lc['view'],lc['span'])
    oil=rgb.copy()
    if water is not None:
        w=np.zeros((128,)*3,bool);w[CROP]=water[CROP]
        wd,wn,wc=geometry_buffer(mesh(w,2),c)
        visible=(wc==2)&(wd>=depth)
        shade=.74+.26*np.maximum(wn@LIGHT,0)
        wrgb=PALETTE[2][None,None,:]*shade[...,None]
        rgb[visible]=.88*rgb[visible]+.12*wrgb[visible]
    if ghost is not None:
        gray=ghost@np.array([.2126,.7152,.0722],np.float32)
        faint=1-.17*(1-gray)
        bg=categories==0;rgb[bg]=faint[bg,None]
    result={'rgb':np.clip(rgb,0,1),'oil':oil,'camera':c,'depth':depth}
    del m,normals,categories,shadow
    gc.collect()
    return result

def frame(ax,c,depth):
    corners=np.array(list(itertools.product((31.5,95.5),repeat=3)))
    for i in range(8):
        for j in range(i):
            if np.count_nonzero(corners[i]!=corners[j])!=1:continue
            xyz=corners[i]+np.linspace(0,1,240)[:,None]*(corners[j]-corners[i])
            q=project(xyz,c)
            ix=np.clip(q[:,0].astype(int),0,c['width']-1)
            iy=np.clip(q[:,1].astype(int),0,c['height']-1)
            visible=q[:,2]>=depth[iy,ix]-.25
            xy=q[:,:2].copy();xy[~visible]=np.nan
            ax.plot(xy[:,0],xy[:,1],lw=.4,color='#CED6DE')

def heading(ax,letter,text,y=1.03):
    ax.text(.0,y,letter,transform=ax.transAxes,weight='bold',fontsize=10,
            ha='left',va='bottom',color=INK)
    # Panel meanings are stated in the manuscript caption, not inside the image.

def image_panel(fig,bounds,result,letter,title):
    ax=fig.add_axes(bounds,label=letter)
    ax.imshow(result['rgb'],interpolation='none')
    frame(ax,result['camera'],result['depth'])
    ax.set(xlim=(-.5,result['camera']['width']-.5),
           ylim=(result['camera']['height']-.5,-.5))
    ax.set_axis_off();heading(ax,letter,title)
    return ax

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

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

def render(out,SOURCE=SOURCE,TRAJECTORY=TRAJECTORY):
    if sys.platform=='win32':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':STEM,
        'image.composite_image':False,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7,'legend.frameon':False})
    with np.load(SOURCE,allow_pickle=False) as d:
        initial=d['initial_labels'];truth=d['truth_labels_a'];prob=d['prediction_a'];steps=d['steps']
    metadata=json.loads(SOURCE.with_suffix('.source.json').read_text(encoding='utf-8'))
    assert metadata['pair_id']=='z0_y256_x256|NPI-012|P1-22-DEV-A|S3~S5'
    assert metadata['schedules_evaluated']==['S5x30 -> S3x70']
    assert initial.shape==(128,)*3 and steps.tolist()==[0,25,50,75,100]
    assert not (initial[CROP]==0).any() and np.isfinite(prob).all()
    oil=initial>=3;n0=int(oil[CROP].sum());changed=oil&(truth[-1]==2)
    assert n0==35740 and int(changed[CROP].sum())==17087
    assert np.all(prob[:,~oil]==0) and prob.min()>=0 and prob.max()<=1
    cohorts=contact_followup(initial,truth,prob,TRAJECTORY)
    print(json.dumps({'contact_followup':cohorts}),flush=True)
    final=(prob[-1]>=.5)&oil
    scenes=[]
    print('Rendering central initial state',flush=True)
    scenes.append(volume(oil,initial==2))
    print('Rendering central simulator and prediction at step 100',flush=True)
    scenes.append(volume(truth[-1]>=3,truth[-1]==2))
    scenes.append(volume(final,(initial==2)|(oil&~final)))
    print('Rendering the converted interior oil',flush=True)
    scenes.append(volume(changed,changed=True,ghost=scenes[0]['oil']))
    crop_truth=truth[(slice(None),)+CROP]>=3
    crop_prob=prob[(slice(None),)+CROP].astype(np.float64)
    true_mob=np.array([np.count_nonzero(oil[CROP]&~t) for t in crop_truth])
    pred_mob=n0-crop_prob.sum((1,2,3))
    edges=np.arange(32,97,16)
    true_depth=[];pred_depth=[];den=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        c=(slice(32,96),slice(32,96),slice(int(lo),int(hi)))
        n=int(oil[c].sum());den.append(n)
        true_depth.append(int(changed[c].sum()))
        pred_depth.append(float(n-prob[-1][c].astype(np.float64).sum()))
    assert sum(true_depth)==17087 and sum(den)==n0
    assert np.isclose(sum(pred_depth),pred_mob[-1],rtol=0,atol=1e-8)
    fig=plt.figure(figsize=(WIDTH/25.4,HEIGHT/25.4),facecolor='white')
    xs=[.018,.348,.678];w=.307;h=w*WIDTH/(HEIGHT*1.06)
    first=.534;second=.052
    axes=[]
    for k,title in enumerate(('Initial · step 0','Simulator · step 100','Prediction · step 100')):
        axes.append(image_panel(fig,[xs[k],first,w,h],scenes[k],'abc'[k],title))
    axes.append(image_panel(fig,[xs[0],second,w,h],scenes[3],'d','Mobilized interior oil'))
    plots=[]
    for i in range(2):
        ax=fig.add_axes([.402+i*.330,.158,.252,.310]);plots.append(ax)
        heading(ax,'ef'[i],('Interior mobilization','Mobilization: steps 51–75')[i],y=1.075)
        ax.tick_params(labelsize=7.5,length=3,pad=2)
        ax.grid(axis='y',color='#E6EBEF',lw=.5,zorder=0)
    a,b=plots
    a.plot(steps,100*true_mob/n0,color=GREEN,marker='o',ms=3.6,lw=1.35,label='Simulator')
    a.plot(steps,100*pred_mob/n0,color=PINK,marker='s',ms=3.6,lw=1.35,ls='--',label='Prediction')
    a.set(xlim=(-2,132),ylim=(0,65),xticks=[0,50,100],yticks=[0,20,40,60],
          xlabel='Simulator step',ylabel='Initial local oil mobilized (%)')
    for value in (100*true_mob[-1]/n0,100*pred_mob[-1]/n0):
        a.annotate(f'{value:.2f}%',xy=(100,value),xytext=(4,0),
                   textcoords='offset points',ha='left',va='center',
                   fontsize=6.8,color=INK,zorder=5)
    x=np.arange(2);bw=.32
    truth_contact=[r['simulator_mobilized_percent'] for r in cohorts[:2]]
    pred_contact=[r['prediction_mobilized_percent'] for r in cohorts[:2]]
    bars_true=b.bar(x-bw/2,truth_contact,bw,color=GREEN,edgecolor='white',lw=.3,zorder=3)
    bars_pred=b.bar(x+bw/2,pred_contact,bw,facecolor='white',edgecolor=PINK,lw=1.,zorder=3)
    b.set(xlim=(-.6,1.6),ylim=(0,100),xticks=x,
          xticklabels=['Contact','No contact'],yticks=[0,50,100],
          xlabel='Water contact at step 50',ylabel='Mobilized during steps 51–75 (%)')
    for bars,alignment in ((bars_true,'right'),(bars_pred,'left')):
        for bar in bars:
            value=bar.get_height()
            b.annotate(f'{value:.1f}%',xy=(bar.get_x()+bar.get_width()/2,value),
                       xytext=(0,4),textcoords='offset points',ha=alignment,va='bottom',
                       fontsize=6.8,color=INK,zorder=5)
    b.tick_params(axis='x',labelsize=7)
    legend=[Line2D([],[],color=GREEN,lw=1.4,marker='o',ms=3,label='Simulator'),
            Line2D([],[],color=PINK,lw=1.4,ls='--',marker='s',ms=3,label='Prediction')]
    fig.legend(handles=legend,loc='center',bbox_to_anchor=(.663,.073),ncol=2,fontsize=7.5)
    fig.legend(handles=[Patch(facecolor=PALETTE[7],label='Remaining oil'),
        Patch(facecolor='#CDE5F3',label='Water'),Patch(facecolor=PALETTE[6],label='Mobilized oil')],
        loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,fontsize=8,handlelength=1)
    prefix=out/STEM
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=list('abcd'),
        row_groups=[list('abc')],column_groups=[list('ad')],
        json_out=str(prefix)+'.alignment.json',strict=True,require_panel_labels=True)
    require_matplotlib_panel_alignment(fig,axes=plots,panel_ids=list('ef'),row_groups=[list('ef')],
        json_out=str(prefix)+'.plots-alignment.json',strict=True,require_panel_labels=True)
    fig.savefig(str(prefix)+'.pdf',dpi=600,metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(str(prefix)+'.svg',dpi=600,metadata={'Date':None})
    fig.savefig(str(prefix)+'.png',dpi=300)
    fig.savefig(str(prefix)+'.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    write_csv(Path(str(prefix)+'.time.csv'),[
        {'step':int(s),'initial_oil':n0,'simulator_mobilized_voxels':int(t),
         'predicted_expected_mobilized_voxels':float(p),'simulator_percent':100*float(t)/n0,
         'prediction_percent':100*float(p)/n0} for s,t,p in zip(steps,true_mob,pred_mob)])
    write_csv(Path(str(prefix)+'.depth.csv'),[
        {'x_start':int(lo),'x_stop_exclusive':int(hi),'initial_slab_oil':int(n),
         'simulator_mobilized_voxels':int(t),'predicted_expected_mobilized_voxels':float(p),
         'simulator_percent':100*float(t)/n,'prediction_percent':100*float(p)/n}
        for lo,hi,n,t,p in zip(edges[:-1],edges[1:],den,true_depth,pred_depth)])
    write_csv(Path(str(prefix)+'.contact.csv'),cohorts)
    record={'source':SOURCE.name,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'parent_pair':metadata['pair_id'],'parent_shape':[128]*3,'crop_zyx':[[32,96]]*3,
        'global_stack_crop_zyx':[[32,96],[288,352],[288,352]],
        'crop_selection':'Geometric central half along each axis; independent of local prediction errors.',
        'parent_selection':metadata['initial_geometry_selection'],
        'distance_from_each_parent_face_voxels':32,'exterior_background_voxels':0,
        'initial_local_oil':n0,'true_mobilized_voxels':int(true_mob[-1]),
        'true_mobilized_percent':float(100*true_mob[-1]/n0),
        'predicted_expected_mobilized_percent':float(100*pred_mob[-1]/n0),
        'geometry':'Native 0.5 marching cubes, smooth-normal lighting; no geometric smoothing/decimation.',
        'display_crop':'All four 3-D scenes share this central crop, scale and camera.',
        'scene_settings':{'camera':CAMERA,'water_opacity':.12,'oil_opacity':1,'rock':'hidden'},
        'prediction_surface_level':.5,'numerical_counts':'Continuous probabilities, not thresholded.',
        'time_plot':'Five cached time points connected as guides, not interpolated observations.',
        'depth_archive':'Four non-overlapping x slabs; retained as source data, not repeated in this figure.',
        'contact_plot':{
            'baseline_step':50,'followup_step':75,'unchanged_action':'S3',
            'trajectory':TRAJECTORY.name,
            'trajectory_sha256':hashlib.sha256(TRAJECTORY.read_bytes()).hexdigest(),
            'cohort_definition':'Simulator-remaining oil at step 50 in inlet-connected pores, split by face-sharing water contact on the full 128^3 grid before cropping.',
            'denominator':'Number of simulator-remaining oil sites in the SAME baseline group, for both simulator and prediction.',
            'prediction_numerator':'Sum of continuous p50 minus p75 in each baseline group, without reinitialization at step 50.',
            'not_shown':'The 1973 remaining oil sites outside the inlet-connected pore component had zero simulator and predicted conversion.',
            'interpretation':'Descriptive single-case temporal localization; groups differ in geometry and initial classes. Not a contact intervention or independent replicate comparison.',
            'cohorts':cohorts},
        'units':'Voxel indices and dimensionless fractions; no physical length calibration.',
        'replication':'One existing DEV case and one frozen seed; no uncertainty intervals.',
        'new_inference':False,'new_simulation':False,'test_accessed':False,
        'physical_size_mm':[WIDTH,HEIGHT],
        'crop_explanation':'Coordinates and parent-volume context are in the figure caption, not an in-image explanation strip.',
        'relation_to_supplement':'Main Figure 5 is the central detail of the complete parent in Supplementary Figure S2; the parent axial profile remains in that supplement.'}
    Path(str(prefix)+'.source.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'CENTRAL64_RENDER_COMPLETE','true_mobilized_percent':record['true_mobilized_percent'],
                     'predicted_percent':record['predicted_expected_mobilized_percent'],'prefix':str(prefix)}),flush=True)

if __name__=='__main__':
    raise SystemExit("Use reproduce_3d.py to validate inputs and protect existing outputs.")
