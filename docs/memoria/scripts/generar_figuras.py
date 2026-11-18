#!/usr/bin/env python3
"""Genera las figuras y métricas de la memoria.

Uso: python3 generar_figuras.py <dir_datos> <dir_figuras>
<dir_datos> contiene hba-tum.txt, <ejecucion>_traj.csv, <ejecucion>_stats.json y
mapas submuestreados sem.npz, color.npz, yolo.npz (xyz, rgb, l, c) extraídos de los .ply.
"""
import json, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
D=sys.argv[1]; O=sys.argv[2]
plt.rcParams.update({'font.size':8,'font.family':'serif','axes.linewidth':0.6,'savefig.dpi':300})
CLASSES=['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle']
ES={'road':'calzada','sidewalk':'acera','building':'edificio','wall':'muro','fence':'valla','pole':'poste','traffic light':'semáforo','traffic sign':'señal','vegetation':'vegetación','terrain':'terreno','car':'coche','truck':'camión','bus':'autobús','train':'tren','motorcycle':'moto','bicycle':'bicicleta'}
PAL=np.array([[128,64,128],[244,35,232],[214,140,60],[102,102,156],[190,153,153],[153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],[70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],[0,60,100],[0,80,100],[0,0,230],[119,11,32]])/255.
def lut(l):
    c=np.full((len(l),3),0.78); m=l<19; c[m]=PAL[l[m]]; return c
# --- trayectorias y alineación (Umeyama) ---
def umeyama(src,dst):
    ms,md=src.mean(0),dst.mean(0); S=(dst-md).T@(src-ms)/len(src); U,_,Vt=np.linalg.svd(S)
    Dm=np.eye(3); Dm[2,2]=-1 if np.linalg.det(U@Vt)<0 else 1; R=U@Dm@Vt; return R,md-R@ms
gt=np.loadtxt(f'{D}/hba-tum.txt'); tg,pg=gt[:,0],gt[:,1:4]
def assoc(te,pe):
    idx=np.clip(np.searchsorted(te,tg),1,len(te)-1); l=te[idx-1]; r=te[idx]
    idx=np.where(np.abs(tg-l)<np.abs(tg-r),idx-1,idx); ok=np.abs(te[idx]-tg)<=0.05
    return pg[ok],pe[idx[ok]],tg[ok]
res={}
runs=[('output_color_v1','Color'),('output_semantic_v1','Semántico v1'),('output_semantic_v2','Semántico v2'),('output_semantic_v3','Semántico v3'),('output_yolo_v1','YOLO')]
for r,_ in runs:
    a=np.loadtxt(f'{D}/{r}_traj.csv',delimiter=',',skiprows=1); G,E,T=assoc(a[:,0],a[:,1:4])
    R,t=umeyama(E,G); Ea=E@R.T+t; e=np.linalg.norm(Ea-G,axis=1)
    dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(G,axis=0),axis=1))]
    # RPE traslacional sobre tramos de 10 m
    rp=[]
    for i in range(len(G)):
        j=np.searchsorted(dist,dist[i]+10.0)
        if j>=len(G): break
        rp.append(np.linalg.norm((Ea[j]-Ea[i])-(G[j]-G[i])))
    rp=np.array(rp)
    res[r]=dict(n=len(e),total=len(tg),L=dist[-1],rmse=np.sqrt((e**2).mean()),mean=e.mean(),med=np.median(e),max=e.max(),
                rmse_xy=np.sqrt((np.linalg.norm((Ea-G)[:,:2],axis=1)**2).mean()),rmse_z=np.sqrt((((Ea-G)[:,2])**2).mean()),
                final=e[-1],rpe_mean=rp.mean(),rpe_rmse=np.sqrt((rp**2).mean()),rpe_pct=100*rp.mean()/10,
                tilt=np.degrees(np.arccos(np.clip(R[2,2],-1,1))),R=R.tolist(),t=t.tolist())
    if r=='output_semantic_v3': best=(G,Ea,e,dist,R,t)
json.dump(res,open(f'{O}/../metricas.json','w'),indent=1)
for r,n in runs: print(n, {k:round(v,3) for k,v in res[r].items() if not isinstance(v,list)})
# --- Fig ATE ---
G,Ea,e,dist,Rg,tg_=best
fig,ax=plt.subplots(1,2,figsize=(7.0,2.6),gridspec_kw={'width_ratios':[1,1.5]})
ax[0].plot(G[:,0],G[:,1],color='0.35',lw=2.2,label='Referencia (HBA)'); ax[0].plot(Ea[:,0],Ea[:,1],color='#c0392b',lw=0.8,label='RKO-LIO')
ax[0].plot(G[0,0],G[0,1],'o',ms=4,color='k'); ax[0].set_aspect('equal'); ax[0].set_xlabel('x (m)'); ax[0].set_ylabel('y (m)'); ax[0].legend(frameon=False,fontsize=7,loc='lower right'); ax[0].set_title('(a) Planta',fontsize=8)
ax[1].plot(dist,e*100,color='#c0392b',lw=0.7); ax[1].axhline(res['output_semantic_v3']['rmse']*100,color='k',ls='--',lw=0.7,label=f"RMSE = {res['output_semantic_v3']['rmse']*100:.1f} cm")
ax[1].set_xlabel('Distancia recorrida (m)'); ax[1].set_ylabel('Error absoluto (cm)'); ax[1].legend(frameon=False,fontsize=7); ax[1].set_xlim(0,dist[-1]); ax[1].set_ylim(0,None); ax[1].set_title('(b) ATE a lo largo del recorrido',fontsize=8)
for a_ in ax: a_.grid(alpha=0.25,lw=0.4)
fig.tight_layout(); fig.savefig(f'{O}/ate.pdf'); plt.close()
# --- mapas: transformar al frame de la referencia (nivelado con la gravedad) ---
def tomap(name):
    z=np.load(f'{D}/{name}.npz'); xyz=z['xyz'].astype(np.float64)@Rg.T+tg_
    return xyz,z['rgb'],z['l'],z['c'].astype(np.float32)
def top(ax,xyz,col,order=None,s=0.15):
    o=np.argsort(xyz[:,2]) if order is None else order
    ax.scatter(xyz[o,0],xyz[o,1],c=col[o],s=s,lw=0,rasterized=True); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
xs,rgs,ls,cs=tomap('sem'); xc,rgc,lc,_=tomap('color'); xy,_,ly,_=tomap('yolo')
lim=lambda ax: (ax.set_xlim(np.percentile(xs[:,0],[0.5,99.5])+[-3,3]), ax.set_ylim(np.percentile(xs[:,1],[0.5,99.5])+[-3,3]))
# fig mapa color + semántico
fig,ax=plt.subplots(1,2,figsize=(7.0,3.3))
m=xc[:,2]<np.percentile(xc[:,2],99.5); top(ax[0],xc[m],rgc[m]/255.); lim(ax[0]); ax[0].set_title('(a) Mapa coloreado con la imagen (modo color)',fontsize=8)
ax[0].plot(G[:,0],G[:,1],color='#c0392b',lw=0.6)
top(ax[1],xs,lut(ls)); lim(ax[1]); ax[1].set_title('(b) Mapa semántico (SegFormer)',fontsize=8)
present=[i for i in [0,1,2,4,8,9,13] ]
ax[1].legend(handles=[Patch(color=PAL[i],label=ES[CLASSES[i]]) for i in present],loc='lower left',fontsize=6,frameon=True,ncol=2,handlelength=1,borderpad=0.3)
fig.tight_layout(); fig.savefig(f'{O}/mapas_planta.pdf'); plt.close()
# fig corte lateral (franja central)
c0=np.median(G,axis=0); d=np.array([np.cos(0),np.sin(0),0])
# eje principal del patio: usar PCA de la trayectoria en xy
P=G[:,:2]-G[:,:2].mean(0); _,_,vt=np.linalg.svd(P); u=vt[0]; v=np.array([-u[1],u[0]])
q=xs[:,:2]-G[:,:2].mean(0); a1=q@u; a2=q@v
band=np.abs(a2)<1.5
fig,ax=plt.subplots(figsize=(7.0,2.0))
ax.scatter(a1[band],xs[band,2],c=lut(ls[band]),s=0.8,lw=0,rasterized=True)
ax.set_aspect('equal'); ax.set_xlabel('Posición a lo largo del patio (m)'); ax.set_ylabel('z (m)'); ax.grid(alpha=0.25,lw=0.4)
fig.tight_layout(); fig.savefig(f'{O}/corte.pdf'); plt.close()
# fig segformer vs yolo
fig,ax=plt.subplots(1,2,figsize=(7.0,3.0))
top(ax[0],xs,lut(ls)); lim(ax[0]); ax[0].set_title('(a) SegFormer: 100 % de voxels etiquetados',fontsize=8)
cy=lut(ly); o=np.argsort(ly!=255,kind='stable'); top(ax[1],xy,cy,order=o,s=0.15)
mm=ly!=255; ax[1].scatter(xy[mm,0],xy[mm,1],c=cy[mm],s=0.6,lw=0,rasterized=True); lim(ax[1]); ax[1].set_title('(b) YOLO11-seg: 0,6 % de voxels etiquetados',fontsize=8)
ax[1].legend(handles=[Patch(color=PAL[i],label=ES[CLASSES[i]]) for i in [13,16]]+[Patch(color=(0.78,)*3,label='sin etiqueta')],loc='lower left',fontsize=6,frameon=True)
fig.tight_layout(); fig.savefig(f'{O}/segformer_vs_yolo.pdf'); plt.close()
# fig distribución de clases (log)
st=json.load(open(f'{D}/output_semantic_v3_stats.json'))['voxels_per_class']; sy=json.load(open(f'{D}/output_yolo_v1_stats.json'))['voxels_per_class']
names=[c for c in CLASSES if c in st or c in sy]
fig,ax=plt.subplots(figsize=(3.3,2.3)); x=np.arange(len(names)); w=0.4
ax.bar(x-w/2,[st.get(c,0) for c in names],w,color=[PAL[CLASSES.index(c)] for c in names],edgecolor='k',lw=0.3,label='SegFormer')
ax.bar(x+w/2,[sy.get(c,0) for c in names],w,color='white',edgecolor='k',lw=0.4,hatch='////',label='YOLO11-seg')
ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels([ES[c] for c in names],rotation=60,ha='right',fontsize=6.5); ax.set_ylabel('Voxels'); ax.legend(frameon=False,fontsize=6.5); ax.grid(axis='y',alpha=0.25,lw=0.4)
fig.tight_layout(); fig.savefig(f'{O}/clases.pdf'); plt.close()
# tabla por clase: % y confianza media (submuestreo 35 %)
tab=[]
for i,c in enumerate(CLASSES):
    m=ls==i
    if m.sum()>0: tab.append((c,st.get(c,0),100*st.get(c,0)/sum(st.values()),float(cs[m].mean())))
json.dump(tab,open(f'{O}/../tabla_clases.json','w'))
for t in tab: print(t)
