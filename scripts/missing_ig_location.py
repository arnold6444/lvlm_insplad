"""Input IG of the frozen missing logit; custom, unvalidated spatial proposals."""
from __future__ import annotations
from collections import Counter
import hashlib
import numpy as np
from PIL import Image, ImageFilter

CONFIG = {'steps':32,'integration':'midpoint','baselines':['blur','mean'],
          'blur_radius_divisor':12,'tensor_size':518,'patch_size':14,
          'component_peak_fraction':0.5,'component_mass_fraction':0.15,
          'max_components':3,'connectivity':8,'positive_pixels_only':True,
          'consensus':'minimum of L1-normalized positive patch maps',
          'canary_atol':1e-5,'canary_rtol':1e-4,
          'completeness_warning_relative':0.1,'completeness_warning_absolute':0.1,
          'center_order':[5,2,8,4,6,1,3,7,9], 'random_namespace':'ig-matched-count42/'}


def geometry(size):
    w,h=size
    if min(w,h)<3:raise ValueError('image too small')
    side=max(w,h)
    return dict(original_size=[w,h],side=side,left=(side-w)//2,top=(side-h)//2,
                tensor_size=CONFIG['tensor_size'],patch_size=CONFIG['patch_size'])


def padded(image):
    g=geometry(image.size);im=Image.new('RGB',(g['side'],g['side']),(124,116,104))
    im.paste(image,(g['left'],g['top']))
    return im,g


def baseline(image,kind):
    if kind=='blur':return image.filter(ImageFilter.GaussianBlur(max(2,min(image.size)/CONFIG['blur_radius_divisor'])))
    if kind=='mean':return Image.new('RGB',image.size,tuple(np.asarray(image,dtype=np.float64).mean((0,1)).round().astype(int).tolist()))
    raise ValueError('unknown IG baseline')


def integrate(forward, target, base, steps):
    """Straight-line midpoint IG. Differentiates input only; no optimizer/weight update."""
    import torch
    if type(steps) is not int or steps<1 or target.shape!=base.shape or target.shape[0]!=1:
        raise ValueError('invalid IG shape/steps')
    if not torch.isfinite(target).all() or not torch.isfinite(base).all():raise ValueError('nonfinite input')
    total=torch.zeros_like(target)
    with torch.enable_grad():
        for i in range(steps):
            point=(base+(target-base)*((i+.5)/steps)).detach().requires_grad_(True)
            value=forward(point)
            if value.numel()!=1 or not torch.isfinite(value).all():raise ValueError('nonfinite scalar logit')
            gradient,=torch.autograd.grad(value,point,create_graph=False,retain_graph=False)
            if not torch.isfinite(gradient).all():raise ValueError('nonfinite gradient')
            total.add_(gradient.detach())
    return ((target-base)*total/steps).detach()


def patch_map(signed, g):
    """Remove square padding before pooling; map patch centers to original coordinates."""
    n=g['tensor_size'];p=g['patch_size'];w,h=g['original_size']
    a=np.asarray(signed)
    if a.dtype!=np.float32 or a.shape!=(n,n) or n%p or not np.isfinite(a).all():raise ValueError('invalid signed attribution')
    xs=((np.arange(n)+.5)*g['side']/n-g['left'])/w
    ys=((np.arange(n)+.5)*g['side']/n-g['top'])/h
    xv=(xs>=0)&(xs<1);yv=(ys>=0)&(ys<1)
    positive=np.maximum(a.astype(np.float64),0)*yv[:,None]*xv[None,:]
    grid=positive.reshape(n//p,p,n//p,p).sum((1,3))
    def centers(values,valid):
        parts=values.reshape(-1,p);mask=valid.reshape(-1,p)
        return np.divide((parts*mask).sum(1),mask.sum(1),out=np.zeros(n//p),where=mask.sum(1)>0)
    return grid,centers(xs,xv),centers(ys,yv)


def cell_of(x,y):
    if not np.isfinite([x,y]).all() or not 0<=x<=1 or not 0<=y<=1:raise ValueError('invalid original-frame center')
    # Existing box-center convention: exact thirds belong to the center row/column.
    col=0 if x<1/3 else 2 if x>2/3 else 1
    row=0 if y<1/3 else 2 if y>2/3 else 1
    return row*3+col+1


def components(values,xs,ys):
    values=np.asarray(values,dtype=np.float64)
    if values.shape!=(len(ys),len(xs)) or not np.isfinite(values).all() or (values<0).any():raise ValueError('invalid patch scores')
    if values.max(initial=0)<=0:return []
    active=values>=CONFIG['component_peak_fraction']*values.max()
    denominator=float(values[active].sum());seen=set();found=[]
    for y,x in zip(*np.nonzero(active)):
        if (y,x) in seen:continue
        queue=[(int(y),int(x))];seen.add((y,x));pixels=[]
        while queue:
            yy,xx=queue.pop();pixels.append((yy,xx))
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    ny,nx=yy+dy,xx+dx
                    if (0<=ny<len(ys) and 0<=nx<len(xs) and active[ny,nx] and (ny,nx) not in seen):
                        seen.add((ny,nx));queue.append((ny,nx))
        mass=sum(values[yy,xx] for yy,xx in pixels)
        if mass/denominator<CONFIG['component_mass_fraction']:continue
        cx=sum(xs[xx]*values[yy,xx] for yy,xx in pixels)/mass
        cy=sum(ys[yy]*values[yy,xx] for yy,xx in pixels)/mass
        found.append(dict(cell=cell_of(cx,cy),center_xy=[float(cx),float(cy)],mass=float(mass),
                          retained_mass_fraction=float(mass/denominator),patches=[list(z) for z in sorted(pixels)]))
    return sorted(found,key=lambda r:(-r['mass'],r['center_xy'][1],r['center_xy'][0]))[:CONFIG['max_components']]


def proposals(raw_maps,g):
    maps={};out={};coords=None
    for kind in CONFIG['baselines']:
        value,xs,ys=patch_map(raw_maps[kind],g);coords=(xs,ys)
        maps[kind]=value/max(float(value.sum()),1e-30)
        out[kind]=components(maps[kind],xs,ys)
    maps['consensus']=np.minimum(maps['blur'],maps['mean'])
    out['consensus']=components(maps['consensus'],*coords)
    cells=lambda items:sorted({c['cell'] for c in items}) or None
    return dict(blur_top1=cells(out['blur'][:1]),mean_top1=cells(out['mean'][:1]),
                consensus_top1=cells(out['consensus'][:1]),consensus_multi=cells(out['consensus']),components=out),maps


def matched(core,k,old_map):
    if type(k) is not int or not 0<=k<=CONFIG['max_components']:raise ValueError('invalid matched cardinality')
    if not k:return dict(occlusion_matched_k=None,center_matched_k=None,random_matched_k=None)
    rank=old_map['proposal']['agreement_rank']
    if len(rank)!=9 or not np.isfinite(rank).all():raise ValueError('invalid prior occlusion rank')
    old=sorted(range(1,10),key=lambda c:(-rank[c-1],c))[:k]
    random=sorted(range(1,10),key=lambda c:hashlib.sha256((CONFIG['random_namespace']+core+'/'+str(c)).encode()).hexdigest())[:k]
    return dict(occlusion_matched_k=sorted(old),center_matched_k=sorted(CONFIG['center_order'][:k]),random_matched_k=sorted(random))


class NativeAttributor:
    """Pinned DINO preprocessing/model; FP32 input gradients, frozen parameters."""
    def __init__(self,head):
        import torch
        import run_missing_component_probe as probe
        self.torch=torch;self.engine=probe.DinoFeatures();self.head=head
        if any(p.is_floating_point() and p.dtype!=torch.float32 for p in self.engine.model.parameters()):
            raise ValueError('native FP32 weights required')
        self.stats=Counter(forward_passes=0,backward_passes=0,model_loads=1)
        self.mean=torch.tensor(head['mean'],device='cuda',dtype=torch.float64)
        self.scale=torch.tensor(head['scale'],device='cuda',dtype=torch.float64)
        self.coef=torch.tensor(head['coef'],device='cuda',dtype=torch.float64)

    def tensor(self,image):
        im,g=padded(image)
        values=self.engine.processor(images=im,return_tensors='pt',do_center_crop=False,
                                     size={'height':518,'width':518})['pixel_values'].to('cuda',dtype=self.torch.float32)
        if values.shape!=(1,3,518,518):raise ValueError('changed DINO input shape')
        return values,g

    def cls(self,x):
        self.stats['forward_passes']+=1
        raw=self.engine.model(pixel_values=x).last_hidden_state[:,0].float()
        return self.torch.nn.functional.normalize(raw,p=2,dim=-1)

    def logit(self,x):
        return (((self.cls(x).double()-self.mean)/self.scale)*self.coef).sum()+self.head['intercept']

    def feature(self,image):
        x,_=self.tensor(image)
        with self.torch.no_grad():return self.cls(x)[0].cpu().numpy()

    def attribute(self,image,expected):
        torch=self.torch;x,g=self.tensor(image);before=Counter(self.stats)
        with torch.no_grad():
            feature=self.cls(x)[0];target=float((((feature.double()-self.mean)/self.scale)*self.coef).sum()+self.head['intercept'])
        native=feature.cpu().numpy()
        if not np.allclose(native,expected,atol=CONFIG['canary_atol'],rtol=CONFIG['canary_rtol']):raise ValueError('target feature differs from frozen PR68 feature')
        raw={};diagnostics={}
        input_hash=lambda t:hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
        for kind in CONFIG['baselines']:
            b,bg=self.tensor(baseline(image,kind));assert bg==g
            with torch.no_grad():base=float(self.logit(b))
            def forward(point):
                self.stats['backward_passes']+=1
                return self.logit(point)
            attr=integrate(forward,x,b,CONFIG['steps'])
            signed=attr.sum(1)[0].cpu().numpy().astype(np.float32)
            total=float(signed.astype(np.float64).sum());delta=target-base;residual=total-delta
            raw[kind]=signed
            diagnostics[kind]=dict(baseline_logit=base,target_minus_baseline=delta,attribution_sum=total,
                completeness_residual=residual,relative_residual=abs(residual)/max(abs(delta),1e-12),
                completeness_warning=abs(residual)>max(CONFIG['completeness_warning_absolute'],CONFIG['completeness_warning_relative']*abs(delta)),
                baseline_input_sha256=input_hash(b))
        return dict(raw_maps=raw,target_feature=native,geometry=g,target_logit=target,diagnostics=diagnostics,
                    original_input_sha256=input_hash(x),
                    forward_passes=self.stats['forward_passes']-before['forward_passes'],
                    backward_passes=self.stats['backward_passes']-before['backward_passes'])
