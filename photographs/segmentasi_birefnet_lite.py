"""Runs BiRefNet-general-lite (ONNX) on one image and writes its soft foreground mask.

The image is resized to the model input (1024 x 1024), ImageNet-normalized, and run on the CPU with the
memory arena disabled; the output is min-max normalized and resized back to the image size.

Usage: python segmentasi_birefnet_lite.py model.onnx input.png output_mask.png
"""
import sys, numpy as np, onnxruntime as ort
from PIL import Image
model, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
so=ort.SessionOptions(); so.enable_cpu_mem_arena=False; so.enable_mem_pattern=False
so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC; so.intra_op_num_threads=2
s=ort.InferenceSession(model,so,providers=['CPUExecutionProvider'])
inp=s.get_inputs()[0]; H,W=inp.shape[2],inp.shape[3]
im=Image.open(src).convert('RGB'); x=np.asarray(im.resize((W,H),Image.BILINEAR),dtype=np.float32)/255.0
x=(x-np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225]); x=x.transpose(2,0,1)[None].astype(np.float32)
y=s.run(None,{inp.name:x})[0][0,0]
y=1/(1+np.exp(-y)) if y.min()<0 or y.max()>1 else y
y=(y-y.min())/(y.max()-y.min()+1e-8)
Image.fromarray((y*255).astype(np.uint8)).resize(im.size,Image.BILINEAR).save(dst); print('ok',H,W)
