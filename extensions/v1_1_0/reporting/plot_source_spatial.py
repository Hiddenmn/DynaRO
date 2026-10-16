"""Source-core 3D and temporal localization, reusing the original DEV cache.

No inference or simulation. One fixed central64 crop and one original seed.
Same fixed initial-oil geometry, camera and error scale as external cases.
"""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
import plot_external_figures as pf
from plot_central64_displacement import contact_followup

plt,mpl=pf.plt,pf.mpl
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'inputs_3d/dev_parent_prediction.npz'
CROP=(slice(32,96),)*3

def render(out):
    pf.style()
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'pdf.fonttype':42,'svg.fonttype':'none'})
    with np.load(SOURCE,allow_pickle=False) as data:
        initial=data['initial_labels']; truth=data['truth_labels_a']
        prob=data['prediction_a']; steps=data['steps']
    meta=json.loads(SOURCE.with_suffix('.source.json').read_text(encoding='utf-8'))
    assert meta['pair_id']=='z0_y256_x256|NPI-012|P1-22-DEV-A|S3~S5'
    assert steps.tolist()==[0,25,50,75,100]
    oil=initial>=3; n=int(oil[CROP].sum())
    changed=oil&(truth[-1]==2)
    assert n==35740 and int(changed[CROP].sum())==17087
    cohorts=contact_followup(initial,truth,prob)
    error=prob[-1]-(truth[-1]>=3)
    err_cmap=pf.LinearSegmentedColormap.from_list('signed_oil_error',['#377DA0','#F2F2EE','#C16C92'])
    error_palette=err_cmap(np.linspace(0,1,256))[:,:3]
    phase_palette=np.tile(mpl.colors.to_rgb('#D5DADD'),(256,1))
    phase_palette[255]=mpl.colors.to_rgb('#7560A8')
    moved_palette=phase_palette.copy();moved_palette[255]=mpl.colors.to_rgb('#D18452')
    camera=pf.make_camera(1050,az=-125,el=24);camera['span']*=.5
    fig=plt.figure(figsize=(180/25.4,124/25.4))
    images=[]
    scene_fields=(oil,truth[-1]>=3,error,changed)
    titles=('Initial oil','Remaining oil, step 100','DynaRO error, step 100','Mobilized oil, step 100')
    for i,field in enumerate(scene_fields):
        col=i if i<3 else 0; y=.59 if i<3 else .10
        ax=fig.add_axes([.015+col*.33,y,.307,.33],label='abcd'[i])
        palette=error_palette if i==2 else moved_palette if i==3 else phase_palette
        ax.imshow(pf.surface_rgb(initial,field,camera,palette),interpolation='none')
        ax.set_axis_off()
        ax.text(0,1.02,'abcd'[i],transform=ax.transAxes,weight='bold',fontsize=10,va='bottom')
        ax.set_title(titles[i],fontsize=7.7,pad=5)
        images.append(ax)
    cax=fig.add_axes([.71,.565,.25,.012])
    cb=fig.colorbar(mpl.cm.ScalarMappable(norm=pf.Normalize(-1,1),cmap=err_cmap),
        cax=cax,orientation='horizontal',ticks=(-1,0,1))
    cb.outline.set_linewidth(.4);cb.ax.tick_params(labelsize=6.5,pad=1)
    fig.text(.835,.515,'Prediction − target',ha='center',fontsize=6.8)
    curves=[]
    for i in range(2):
        ax=fig.add_axes([.402+i*.33,.16,.252,.29],label='ef'[i])
        ax.text(-.16,1.08,'ef'[i],transform=ax.transAxes,weight='bold',fontsize=10,va='bottom')
        ax.tick_params(labelsize=6.8)
        curves.append(ax)
    a,b=curves
    true_mob=100*np.array([np.count_nonzero(oil[CROP]&(t[CROP]<3)) for t in truth])/n
    pred_mob=100*(n-prob[(slice(None),)+CROP].astype(np.float64).sum((1,2,3)))/n
    a.plot(steps,true_mob,color=pf.INK,marker='o',ms=3,lw=1.3)
    a.plot(steps,pred_mob,color=pf.COLORS['dynaro'],marker='s',ms=3,lw=1.4,ls='--')
    a.set(xlim=(-2,103),ylim=(0,65),xticks=(0,50,100),yticks=(0,20,40,60),
        xlabel='Displacement step',ylabel='Initial oil mobilized (%)')
    a.set_title('Interior mobilization',fontsize=7.7,pad=9)
    x=np.arange(2);w=.32
    bars0=b.bar(x-w/2,[v['simulator_mobilized_percent'] for v in cohorts[:2]],w,
        color=pf.INK)
    bars1=b.bar(x+w/2,[v['prediction_mobilized_percent'] for v in cohorts[:2]],w,
        fc='white',ec=pf.COLORS['dynaro'],lw=1)
    b.set(xlim=(-.6,1.6),ylim=(0,100),xticks=x,xticklabels=['Contact','No contact'],
        yticks=(0,50,100),xlabel='Water contact at step 50',ylabel='Mobilized in steps 51–75 (%)')
    b.set_title('Contact and subsequent loss',fontsize=7.7,pad=9)
    for bars in (bars0,bars1):
        for bar in bars:
            b.annotate(f'{bar.get_height():.1f}',(bar.get_x()+bar.get_width()/2,bar.get_height()),
                xytext=(0,3),textcoords='offset points',ha='center',va='bottom',fontsize=6.4)
    fig.legend(handles=[pf.Line2D([],[],color=pf.INK,lw=1.3,marker='o',ms=3,label='Target'),
        pf.Line2D([],[],color=pf.COLORS['dynaro'],ls='--',lw=1.3,marker='s',ms=3,label='DynaRO')],
        loc='center',bbox_to_anchor=(.69,.069),frameon=False,ncol=2,fontsize=7)
    fig.legend(handles=[pf.Patch(fc='#7560A8',label='Oil remaining'),pf.Patch(fc='#D5DADD',label='Initial-oil context'),
        pf.Patch(fc='#D18452',label='Oil mobilized')],loc='lower center',bbox_to_anchor=(.50,.005),
        frameon=False,ncol=3,fontsize=7,handlelength=1)
    fig.canvas.draw()
    fig.savefig(out/'figure_source_spatial.pdf',dpi=600,metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(out/'figure_source_spatial.svg',dpi=600)
    fig.savefig(out/'figure_source_spatial.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(out/'figure_source_spatial.png',dpi=450)
    plt.close(fig)
    pf.write_csv(out/'figure_source_spatial_time.csv',[
        dict(step=int(t),target_mobilized_percent=float(v),dynaro_mobilized_percent=float(p),
             initial_oil_sites=n) for t,v,p in zip(steps,true_mob,pred_mob)])
    pf.write_csv(out/'figure_source_spatial_contact.csv',cohorts)
    (out/'figure_source_spatial_case.json').write_text(json.dumps({
        'source':str(SOURCE),'pair_id':meta['pair_id'],'seed':20260811,
        'parent_selection':meta['initial_geometry_selection'],'crop_zyx':[[32,96]]*3,
        'new_inference':False,'new_simulation':False,'initial_oil_sites':n,
        'camera':{'azimuth':-125,'elevation':24},
        'surfaces':'Fixed initial-oil support; nearest inside-voxel field colour; no geometry smoothing.',
        'target_mobilized_percent':float(true_mob[-1]),'dynaro_mobilized_percent':float(pred_mob[-1]),
        'units':'Voxel coordinates and dimensionless fractions; no physical length calibration.'
    },indent=2),encoding='utf-8')

if __name__=='__main__':
    out=HERE/'figures';out.mkdir(exist_ok=True);render(out)
