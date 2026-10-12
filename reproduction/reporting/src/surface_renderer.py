"""Native-grid surfaces and deterministic lighting reused without numerical edits."""
from __future__ import annotations
import itertools
import math
import numpy as np
from numba import njit, prange
from skimage.measure import marching_cubes
from matplotlib.colors import to_rgb

PALETTE = np.asarray([to_rgb(c) for c in (
    "#FFFFFF", "#969696", "#56B4E9", "#E6AD15",
    "#009E73", "#CC79A7", "#D55E00", "#7560A8")], np.float32)
LIGHT = np.asarray([-.6, -.35, .72], np.float32)
LIGHT /= np.linalg.norm(LIGHT)

def make_camera(pixels, az=-55., el=27., direction=None):
    if direction is None:
        a,e=np.deg2rad([az,el])
        view=np.asarray([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)])
    else:
        view=np.asarray(direction,dtype=float)
        view/=np.linalg.norm(view)
    right=np.cross(np.asarray([0.,0.,1.]),view)
    right/=np.linalg.norm(right)
    up=np.cross(view,right)
    centre=np.asarray([63.5]*3)
    corners=np.asarray(list(itertools.product((-.5,127.5),repeat=3)))
    projected=np.c_[(corners-centre)@right,(corners-centre)@up]
    aspect=1.06
    height=max(np.ptp(projected[:,1]),np.ptp(projected[:,0])/aspect)*1.10
    return {"center":centre.astype(np.float32),"right":right.astype(np.float32),
            "up":up.astype(np.float32),"view":view.astype(np.float32),
            "span":np.asarray([height*aspect,height],np.float32),
            "width":int(round(pixels*aspect)),"height":pixels}


def project(xyz,c):
    p=xyz-c["center"]
    x=((p@c["right"])/c["span"][0]+.5)*c["width"]
    y=(.5-(p@c["up"])/c["span"][1])*c["height"]
    z=p@c["view"]
    return np.c_[x,y,z].astype(np.float32)


@njit
def buffers(projected,faces,normals,categories,width,height):
    depth=np.full((height,width),-1e9,np.float32)
    nmap=np.zeros((height,width,3),np.float32)
    cmap=np.zeros((height,width),np.uint8)
    for f in range(len(faces)):
        i0,i1,i2=faces[f]
        p0,p1,p2=projected[i0],projected[i1],projected[i2]
        xmin=max(0,int(np.floor(min(p0[0],p1[0],p2[0]))))
        xmax=min(width-1,int(np.ceil(max(p0[0],p1[0],p2[0]))))
        ymin=max(0,int(np.floor(min(p0[1],p1[1],p2[1]))))
        ymax=min(height-1,int(np.ceil(max(p0[1],p1[1],p2[1]))))
        determinant=(p1[1]-p2[1])*(p0[0]-p2[0])+(p2[0]-p1[0])*(p0[1]-p2[1])
        if abs(determinant)<1e-10:
            continue
        for y in range(ymin,ymax+1):
            for x in range(xmin,xmax+1):
                a=((p1[1]-p2[1])*(x+.5-p2[0])+(p2[0]-p1[0])*(y+.5-p2[1]))/determinant
                b=((p2[1]-p0[1])*(x+.5-p2[0])+(p0[0]-p2[0])*(y+.5-p2[1]))/determinant
                c=1-a-b
                if min(a,b,c)<-1e-6:
                    continue
                z=a*p0[2]+b*p1[2]+c*p2[2]
                if z<=depth[y,x]:
                    continue
                depth[y,x]=z
                n=a*normals[i0]+b*normals[i1]+c*normals[i2]
                length=math.sqrt((n*n).sum())
                if length>1e-8:
                    n/=length
                nmap[y,x]=n
                cmap[y,x]=categories[f]
    return depth,nmap,cmap


@njit
def is_oil(mask,x,y,z):
    ix,iy,iz=int(round(x)),int(round(y)),int(round(z))
    return 0<=ix<128 and 0<=iy<128 and 0<=iz<128 and mask[iz,iy,ix]


@njit(parallel=True)
def light_pixels(depth,nmap,cmap,mask,palette,center,right,up,view,span,
                 light,shadow,lc,lr,lu,lv,lspan):
    height,width=depth.shape
    rgb=np.ones((height,width,3),np.float32)
    mono=rgb.copy()
    # Deterministic local hemisphere rays, not changes to the label geometry.
    angles=np.arange(8)*2*np.pi/8
    half=light+view
    half/=math.sqrt((half*half).sum())
    for y in prange(height):
        for x in range(width):
            if cmap[y,x]==0:
                continue
            n=nmap[y,x]
            point=center+right*((x+.5)/width-.5)*span[0]+up*(.5-(y+.5)/height)*span[1]+view*depth[y,x]
            # Perceptual contact occlusion from actual nearby oil geometry.
            tangent=np.asarray([-n[1],n[0],0.],np.float32)
            length=math.sqrt((tangent*tangent).sum())
            if length<.05:
                tangent=np.asarray([1.,0.,0.],np.float32)
            else:
                tangent/=length
            tangent2=np.cross(n,tangent)
            occluded=0.
            for k in range(8):
                direction=.72*n+.694*(math.cos(angles[k])*tangent+math.sin(angles[k])*tangent2)
                hit=0.
                for distance,weight in ((2.5,1.),(5.,.6),(9.,.35)):
                    p=point+n*.85+direction*distance
                    if is_oil(mask,p[0],p[1],p[2]):
                        hit=max(hit,weight)
                occluded+=hit
            ao=1-.50*occluded/8
            # Softened directional shadow, from the same native oil mesh.
            delta=point-lc
            sx=int(((delta*lr).sum()/lspan[0]+.5)*shadow.shape[1])
            sy=int((.5-(delta*lu).sum()/lspan[1])*shadow.shape[0])
            sd=(delta*lv).sum()
            visibility=0.
            samples=0
            for oy in (-1,0,1):
                for ox in (-1,0,1):
                    xx,yy=sx+ox,sy+oy
                    if 0<=xx<shadow.shape[1] and 0<=yy<shadow.shape[0]:
                        visibility+=1. if sd>=shadow[yy,xx]-.95 else .15
                        samples+=1
            if samples:
                visibility/=samples
            else:
                visibility=1.
            diffuse=max(0.,(n*light).sum())
            fill=max(0.,(n*view).sum())
            brightness=.36*ao+.57*diffuse*visibility+.15*fill
            specular=.055*max(0.,(n*half).sum())**25*visibility
            for channel in range(3):
                rgb[y,x,channel]=min(1.,palette[cmap[y,x],channel]*brightness+specular)
                mono[y,x,channel]=min(1.,palette[7,channel]*brightness+specular)
    return rgb,mono


def mesh(mask, cid):
    vertices, faces, normals, _ = marching_cubes(
        np.pad(mask.astype(np.float32), 1), .5, allow_degenerate=False)
    vertices -= 1
    return {
        "vertices": vertices[:, [2, 1, 0]].astype(np.float32),
        "faces": faces.astype(np.int32),
        "normals": normals[:, [2, 1, 0]].astype(np.float32),
        "categories": np.full(len(faces), cid, np.uint8),
    }


def geometry_buffer(m, camera):
    return buffers(project(m["vertices"], camera), m["faces"], m["normals"],
                   m["categories"], camera["width"], camera["height"])
