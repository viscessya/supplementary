# -*- coding: utf-8 -*-
"""Assembles the photograph figure (Figure 8): (a) photograph, (b) background removed, (c) four-view
input, (d) reconstruction seen from four sides, with arrows between the panels. Panels (c) and (d) are both
2 x 2 grids so that input and result can be compared side by side.

Usage: python susun_figure_pompa_v2.py --asli left.png --bersih left.png --empat <folder with front/left/back/right.png>
           --rekon r45.png r135.png r315.png r225.png --out fig_pompa.png
"""
import argparse, numpy as np
from PIL import Image, ImageDraw, ImageFont
p=argparse.ArgumentParser()
p.add_argument("--asli",required=True); p.add_argument("--bersih",required=True)
p.add_argument("--empat",required=True); p.add_argument("--rekon",nargs=4,required=True); p.add_argument("--out",required=True)
p.add_argument("--font",default="/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf")
a=p.parse_args()
P=560; G=78; F=ImageFont.truetype(a.font,44); g=8
def kotak(path):
    im=Image.open(path).convert("RGB"); w,h=im.size; s=min(w,h)
    return im.crop(((w-s)//2,(h-s)//2,(w-s)//2+s,(h-s)//2+s)).resize((P,P),Image.LANCZOS)
def potong_putih(im,pad=0.06):
    if im.mode=="RGBA":  # render transparan: potong berdasarkan alpha lalu tempel ke putih
        al=np.asarray(im)[...,3]; ys,xs=np.where(al>8)
        w=Image.new("RGB",im.size,"white"); w.paste(im,(0,0),im); im=w
    else:
        a_=np.asarray(im.convert("L")); ys,xs=np.where(a_<245)
    if len(xs)==0: return im
    x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max(); c=max(x1-x0,y1-y0)*(1+2*pad); cx,cy=(x0+x1)/2,(y0+y1)/2
    kan=Image.new("RGB",(int(c),int(c)),"white"); kan.paste(im,(int(c/2-cx),int(c/2-cy))); return kan
def grid(ims,pad):
    k=Image.new("RGB",(P,P),"white"); d=ImageDraw.Draw(k); h=P//2
    for i,im in enumerate(ims):
        t=potong_putih(im,pad).resize((h-2*g,h-2*g),Image.LANCZOS); k.paste(t,((i%2)*h+g,(i//2)*h+g))
    d.line([(h,g),(h,P-g)],fill=(215,215,215),width=2); d.line([(g,h),(P-g,h)],fill=(215,215,215),width=2)
    return k
pa=kotak(a.asli); pb=kotak(a.bersih)
pc=grid([Image.open(f"{a.empat}/{v}.png").convert("RGB") for v in ["front","left","back","right"]],0.06)
pd=grid([Image.open(f).convert("RGBA") for f in a.rekon],0.05)
lab=["(a) Photograph","(b) Background removed","(c) Four-view input","(d) Reconstruction"]
H=P+20+56; W=4*P+3*G
C=Image.new("RGB",(W,H),"white"); d=ImageDraw.Draw(C)
for i,(im,l) in enumerate(zip([pa,pb,pc,pd],lab)):
    x=i*(P+G); C.paste(im,(x,0))
    if i>=2: d.rectangle([x,0,x+P-1,P-1],outline=(150,150,150),width=3)
    tw=d.textlength(l,font=F); d.text((x+(P-tw)/2,P+20),l,fill="black",font=F)
    if i<3:  # panah alur di celah
        cx=x+P+G//2; cy=P//2; L=26
        d.line([(cx-L,cy),(cx+L-10,cy)],fill=(90,90,90),width=6)
        d.polygon([(cx+L+4,cy),(cx+L-16,cy-14),(cx+L-16,cy+14)],fill=(90,90,90))
C.save(a.out); print("OK",C.size)
