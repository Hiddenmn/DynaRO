"""Independent-core evidence: temporal/morphology transfer and fixed 3D cases.

Data: verified saved 30-job summaries and four approved fixed-case forwards.
No training, model execution, sample ranking, smoothing or statistical tests.
SD denotes three training seeds; independent cores are shown separately.
Exports: 180 mm vector PDFs with embedded native-grid raster surface renders.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='2'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import distance_transform_edt
from surface_renderer import mesh, geometry_buffer, make_camera, LIGHT

MODELS=('dynaro','static_front','coarse_front','rtu','fno')
CORES=('WF1','Bentheimer_Core1')
CORE_NAMES={'WF1':'Ketton (WF1)','Bentheimer_Core1':'Bentheimer'}
NAMES=dict(dynaro='DynaRO',static_front='Static front',coarse_front='Regional front',rtu='RTU',fno='FNO')
COLORS=dict(dynaro='#428F7D',static_front='#C97C9D',coarse_front='#8062AA',rtu='#8A8F9E',fno='#AA9B83')
STYLES=dict(dynaro='-',static_front='--',coarse_front=(0,(3,1,1,1)),rtu='-.',fno=':')
SEEDS=('20260811','20260812','20260813')
INK='#293745'

def style():
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7.5,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,
        'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def label(ax,letter):
    ax.text(0,1.045,letter,transform=ax.transAxes,ha='left',va='bottom',
            weight='bold',fontsize=10,color=INK)

def export(fig,out,stem,**alignment):
    fig.canvas.draw()
    fig.savefig(out/(stem+'.pdf'),dpi=600,metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(out/(stem+'.svg'),dpi=600)
    fig.savefig(out/(stem+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(out/(stem+'.png'),dpi=450)
    plt.close(fig)

def quantitative(out):
    source=PROJECT/'external/aggregate.json'
    s=json.loads(source.read_text(encoding='utf-8'))
    data=s['models']; rows=[]
    fig,axs=plt.subplots(2,2,figsize=(180/25.4,94/25.4))
    fig.subplots_adjust(left=.11,right=.985,bottom=.15,top=.86,wspace=.31,hspace=.75)
    for ci,core in enumerate(CORES):
        ax=axs[0,ci]
        for model in reversed(MODELS):
            series=np.array([data[model]['per_seed_summary'][seed]['per_core_equal_block_mean'][core]['100']['site_by_step'] for seed in SEEDS])
            assert series.shape==(3,100) and np.isfinite(series).all()
            mean,sd=series.mean(0),series.std(0,ddof=1)
            ax.fill_between(np.arange(1,101),mean-sd,mean+sd,color=COLORS[model],alpha=.12,lw=0)
            ax.plot(np.arange(1,101),mean,color=COLORS[model],ls=STYLES[model],lw=1.7 if model=='dynaro' else 1.05)
            for i in range(100):
                rows.append(dict(core=core,model=model,metric='site_brier',step=i+1,oil_class='',
                                 seed1=series[0,i],seed2=series[1,i],seed3=series[2,i],mean=mean[i],sd=sd[i]))
        ax.axvline(50,ls=(0,(2,2)),lw=.65,color='#7C878E')
        ax.set(xlim=(0,100),ylim=(0,None),xticks=(0,25,50,75,100),xlabel='Displacement step',ylabel='Site-state Brier error')
        ax.set_title(CORE_NAMES[core],fontsize=8.4,pad=9,weight='semibold')
        label(ax,chr(97+ci))
        ax=axs[1,ci]
        for mi,model in enumerate(MODELS):
            values=np.array([data[model]['per_seed_summary'][seed]['per_core_equal_block_mean'][core]['100']['last_class_inventory_fraction_mae_by_class'] for seed in SEEDS])*100
            assert values.shape==(3,5)
            mean,sd=values.mean(0),values.std(0,ddof=1)
            pos=np.arange(5)+(mi-2)*.14
            ax.errorbar(pos,mean,yerr=sd,fmt='o',ms=3.3 if model=='dynaro' else 2.6,
                        capsize=1.2,lw=.7,color=COLORS[model],zorder=5-mi*.1)
            for ki in range(5):
                rows.append(dict(core=core,model=model,metric='inventory_mae_pp',step=100,oil_class=ki+3,
                    seed1=values[0,ki],seed2=values[1,ki],seed3=values[2,ki],mean=mean[ki],sd=sd[ki]))
        ax.set(xlim=(-.55,4.55),ylim=(0,None),xticks=np.arange(5),
               xticklabels=['Film','Corner','Droplet','Column','Cluster'],
               ylabel='Class inventory MAE (pp)',xlabel='Initial oil category')
        ax.tick_params(axis='x',labelsize=6.8)
        label(ax,chr(99+ci))
    handles=[Line2D([],[],color=COLORS[m],ls=STYLES[m],lw=1.5,label=NAMES[m]) for m in MODELS]
    fig.legend(handles=handles,ncol=5,loc='upper center',bbox_to_anchor=(.53,1),
               frameon=False,handlelength=2.2,columnspacing=1.3,fontsize=7.1)
    export(fig,out,'figure_external_transfer',axes=list(axs.flat),panel_ids=list('abcd'))
    write_csv(out/'figure_external_transfer_source.csv',rows)

def surface_result(initial,scalar,camera):
    # The same initial-oil surface is used for truth and both error fields.
    # Nearest inside-voxel lookup prevents a zero-valued exterior contaminating
    # surface colours. Coordinates and geometry are never smoothed or deformed.
    mask=np.zeros(initial.shape,bool);mask[32:96,32:96,32:96]=initial[32:96,32:96,32:96]>=3
    geo=mesh(mask,1)
    nearest=distance_transform_edt(~mask,return_distances=False,return_indices=True)
    centres=geo['vertices'][geo['faces']].mean(1)[:,[2,1,0]]
    pos=np.clip(np.rint(centres).astype(int),0,127)
    z,y,x=nearest[:,pos[:,0],pos[:,1],pos[:,2]]
    # Quantized colour only, <=1/127 amplitude error; source floats are retained.
    geo['categories']=np.clip(np.rint((scalar[z,y,x]+1)*127)+1,1,255).astype(np.uint8)
    depth,normals,cats=geometry_buffer(geo,camera)
    return depth,normals,cats

def surface_rgb(initial,scalar,camera,palette):
    depth,normals,cats=surface_result(initial,scalar,camera)
    shade=.62+.38*np.maximum(normals@LIGHT,0)
    rgb=palette[cats]*shade[...,None]
    rgb[cats==0]=1
    return np.clip(rgb,0,1)

def fixed_cases(out):
    cases=HERE/'fixed_cases'
    completion=json.loads((cases/'complete.json').read_text(encoding='utf-8'))
    assert completion['status']=='FOUR_FIXED_FIGURE_FORWARDS_COMPLETE'
    err_cmap=LinearSegmentedColormap.from_list('signed_oil_error',['#377DA0','#F2F2EE','#C16C92'])
    error_palette=err_cmap(np.linspace(0,1,256))[:,:3]
    truth_palette=np.tile(np.array(mpl.colors.to_rgb('#D5DADD')),(256,1))
    truth_palette[255]=mpl.colors.to_rgb('#7560A8')
    camera=make_camera(780,az=-125,el=24);camera['span']*=.5
    fig=plt.figure(figsize=(180/25.4,103/25.4))
    images=[];curves=[];rows=[];description=[]
    for ri,core in enumerate(CORES):
        with np.load(cases/(core+'_inputs.npz'),allow_pickle=False) as d:
            initial=d['initial'];removal=d['first_removal_step']
        ids=np.flatnonzero(initial.ravel()>=3)
        roi=np.zeros(initial.shape,bool);roi[32:96,32:96,32:96]=1
        pick=roi.ravel()[ids]
        probs={m:np.load(cases/(core+'_'+m+'.npz'),allow_pickle=False)['remaining_sites'] for m in ('dynaro','static_front')}
        steps=np.array([0,25,50,75,100])
        truth=np.array([(removal==0)|(removal>t) for t in steps],np.float32)
        fields=[]
        for col in range(3):
            field=np.zeros(initial.shape,np.float32)
            field.ravel()[ids]=truth[-1] if col==0 else probs[('dynaro','static_front')[col-1]][-1]-truth[-1]
            fields.append(field)
        ybase=.58 if ri==0 else .12
        for ci,field in enumerate(fields):
            ax=fig.add_axes([.015+ci*.237,ybase,.226,.34],label=chr(97+ri*4+ci))
            rgb=surface_rgb(initial,field,camera,truth_palette if ci==0 else error_palette)
            ax.imshow(rgb,interpolation='none');ax.set_axis_off()
            ax.text(.02,1.015,chr(97+ri*4+ci),transform=ax.transAxes,weight='bold',fontsize=10,va='bottom')
            ax.set_title(('Target, step 100','DynaRO error','Static-front error')[ci],fontsize=7.7,pad=4)
            images.append(ax)
        ax=fig.add_axes([.78,ybase+.04,.205,.26],label=chr(100+ri*4));curves.append(ax)
        for model in ('target','dynaro','static_front'):
            p=truth if model=='target' else np.vstack([np.ones((1,len(ids)),np.float32),probs[model]])
            series=p[:,pick].mean(1)
            ax.plot(steps,series,marker='o',ms=2.8,lw=1.25,color='#293745' if model=='target' else COLORS[model],
                    ls='-' if model!='static_front' else '--')
            for st,val in zip(steps,series):
                rows.append(dict(core=core,model=model,step=int(st),remaining_fraction=float(val),initial_oil_sites=int(pick.sum())))
        ax.set(xlim=(0,100),ylim=(-.02,1.04),xticks=(0,50,100),yticks=(0,.5,1),xlabel='Displacement step',ylabel='Remaining oil fraction')
        ax.tick_params(labelsize=6.8);ax.set_title('Interior inventory',fontsize=7.7,pad=9)
        ax.text(-.21,1.1,chr(100+ri*4),transform=ax.transAxes,weight='bold',fontsize=10,va='bottom')
        fig.text(.015,ybase-.02,CORE_NAMES[core],fontsize=8.6,weight='bold',color=INK)
        description.append(dict(core=core,block=completion['cases'][ri]['block_id'],initial_oil_sites=int(pick.sum()),
                                truth_remaining50=float(truth[2,pick].mean()),truth_remaining100=float(truth[4,pick].mean()),
                                dynaro_remaining100=float(probs['dynaro'][3,pick].mean()),
                                static_remaining100=float(probs['static_front'][3,pick].mean())))
    cax=fig.add_axes([.285,.043,.31,.013])
    cb=fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(-1,1),cmap=err_cmap),cax=cax,orientation='horizontal',ticks=(-1,0,1))
    cb.outline.set_linewidth(.4);cb.ax.tick_params(labelsize=6.5,pad=1)
    fig.text(.44,.085,'Signed occupancy error (prediction - target)',ha='center',fontsize=7)
    fig.legend(handles=[Patch(fc='#7560A8',label='Remaining oil'),Patch(fc='#D5DADD',label='Displaced oil')],
        loc='upper left',bbox_to_anchor=(.005,.084),frameon=False,fontsize=6.8,handlelength=1)
    fig.legend(handles=[Line2D([],[],color=c,lw=1.3,label=n) for c,n in [('#293745','Target'),(COLORS['dynaro'],'DynaRO'),(COLORS['static_front'],'Static front')]],
               loc='center',bbox_to_anchor=(.81,.038),frameon=False,fontsize=6.4,ncol=3,
               handlelength=.9,handletextpad=.3,columnspacing=.6)
    export(fig,out,'figure_external_spatial',axes=images+curves,
        panel_ids=['a','b','c','e','f','g','d','h'],
        row_groups=[['a','b','c'],['e','f','g']],
        column_groups=[['a','e'],['b','f'],['c','g'],['d','h']])
    write_csv(out/'figure_external_spatial_source.csv',rows)
    (out/'fixed_case_interpretation.json').write_text(json.dumps(description,indent=2),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=HERE/'figures')
    parser.add_argument('--only',choices=('quantitative','spatial','both'),default='both')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True);style()
    if args.only in ('both','quantitative'):quantitative(args.out)
    if args.only in ('both','spatial'):fixed_cases(args.out)
