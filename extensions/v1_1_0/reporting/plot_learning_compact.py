"""Reflow the original individual-seed learning-rule gains into one column."""
from pathlib import Path
import plot_external_figures as pf
from plot_seedwise_bars_v2 import load_rows
plt,mpl=pf.plt,pf.mpl
HERE=Path(__file__).resolve().parent

def render(out):
    pf.style()
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'pdf.fonttype':42,'svg.fonttype':'none'})
    rows=load_rows();fig,axes=plt.subplots(2,1,figsize=(86/25.4,100/25.4),sharex=True)
    fig.subplots_adjust(left=.235,right=.97,bottom=.14,top=.85,hspace=.55)
    colors=('#61AC9C','#CE8CA5')
    for ax,letter,split in zip(axes,'ab',('Development','Locked test')):
        values=[v for v in rows if v['split']==split]
        assert [v['seed'] for v in values]==[20260811,20260812,20260813]
        for y,row in enumerate(values):
            for key,dy,color in zip(('site_reduction_pct','response_reduction_pct'),(-.16,.16),colors):
                value=row[key]
                ax.barh(y+dy,value,height=.27,color=color,zorder=3)
                ax.annotate(f'{value:.2f}',(value,y+dy),xytext=(3 if value>=0 else -3,0),
                    textcoords='offset points',ha='left' if value>=0 else 'right',va='center',fontsize=6.7)
        ax.set(yticks=range(3),yticklabels=[str(v['seed']) for v in values],
            ylim=(2.55,-.55),xlim=(-1.6,5.8),xticks=(-1,0,1,2,3,4,5))
        ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=4,labelsize=6.6)
        ax.axvline(0,color='#7E858B',lw=.6)
        ax.set_title('Development · 400 pairs' if letter=='a' else 'Test · 600 pairs',fontsize=7.5,pad=5)
        ax.text(-.25,1.06,letter,transform=ax.transAxes,weight='bold',fontsize=9,va='bottom')
    axes[-1].set_xlabel('Error reduction vs fixed weight 50 (%)',fontsize=7)
    fig.legend(handles=[pf.Patch(fc=c,label=n) for c,n in zip(colors,('Site-state Brier','Paired-response MSE'))],
        loc='upper center',bbox_to_anchor=(.54,.997),frameon=False,fontsize=7,ncol=1)
    fig.canvas.draw()
    fig.savefig(out/'figure_learning_compact.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(out/'figure_learning_compact.svg')
    fig.savefig(out/'figure_learning_compact.png',dpi=450)
    fig.savefig(out/'figure_learning_compact.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig);pf.write_csv(out/'figure_learning_compact_source.csv',rows)
if __name__=='__main__':render(HERE/'figures')
