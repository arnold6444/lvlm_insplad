"""Bounded IG numerical diagnostics; frozen FP32 network, no training or GT."""
from collections import Counter
from contextlib import nullcontext
import hashlib
import numpy as np
import missing_ig_location as ig

CONFIG = dict(cohort_size=4, selection='sha256(ig-numerics-v1/+core) ascending',
              conditions=[['midpoint',32],['midpoint',128],['midpoint',512],['gausslegendre',128]],
              baselines=['blur','mean'], alphas=[.1,.5,.9], epsilons=[.01,.001,.0001],
              backends=['default','sdpa_math'], fd_atol=.1, fd_rtol=.05,
              completeness_atol=.1, completeness_rtol=.1)

def choose(cores):
    return sorted(cores,key=lambda c:hashlib.sha256(('ig-numerics-v1/'+c).encode()).hexdigest())[:CONFIG['cohort_size']]

def quadrature(method,n):
    if type(n) is not int or n<1:raise ValueError('positive integer steps required')
    if method=='midpoint':return (np.arange(n)+.5)/n,np.full(n,1/n)
    if method=='gausslegendre':
        points,weights=np.polynomial.legendre.leggauss(n)
        return (points+1)/2,weights/2
    raise ValueError('unknown quadrature')

def integrate_pair(forward,target,base,method,n):
    """Same FP32 gradients; native FP32 and FP64 accumulation compared together."""
    import torch
    points,weights=quadrature(method,n)
    if target.shape!=base.shape or target.shape[0]!=1 or target.dtype!=torch.float32:
        raise ValueError('FP32 batch-one matching inputs required')
    if not torch.isfinite(target).all() or not torch.isfinite(base).all():raise ValueError('nonfinite input')
    total32=torch.zeros_like(target);total64=torch.zeros_like(target,dtype=torch.float64)
    with torch.enable_grad():
        for alpha,weight in zip(points,weights):
            x=(base+(target-base)*float(alpha)).detach().requires_grad_(True)
            value=forward(x)
            if value.numel()!=1 or not torch.isfinite(value).all():raise ValueError('nonfinite scalar output')
            gradient,=torch.autograd.grad(value,x)
            if not torch.isfinite(gradient).all():raise ValueError('nonfinite gradient')
            # Midpoint FP32 branch exactly preserves PR69's accumulation/order.
            total32.add_(gradient.detach() if method=='midpoint' else gradient.detach()*float(weight))
            total64.add_(gradient.detach().double()*float(weight))
    attr32=(target-base)*total32/n if method=='midpoint' else (target-base)*total32
    attr64=(target-base).double()*total64
    return attr32.detach(),attr64.detach()

def completeness(signed,target,base):
    a=np.asarray(signed)
    if not np.isfinite(a).all():raise ValueError('nonfinite attribution')
    total=float(a.astype(np.float64).sum());delta=float(target-base);residual=total-delta
    return dict(target_minus_baseline=delta,attribution_sum=total,residual=residual,
                relative_residual=abs(residual)/max(abs(delta),1e-12),
                warning=abs(residual)>max(CONFIG['completeness_atol'],CONFIG['completeness_rtol']*abs(delta)))

def similarity(a,b):
    a=np.asarray(a,dtype=float).ravel();b=np.asarray(b,dtype=float).ravel()
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('invalid map pair')
    den=float(np.linalg.norm(a)*np.linalg.norm(b))
    return dict(cosine=float(a@b/den) if den else None,
                relative_l1=float(np.abs(a-b).sum()/max(np.abs(b).sum(),1e-12)))

def directional(forward,target,base,alphas=None,epsilons=None):
    import torch
    result=[];direction=target-base
    with torch.enable_grad():
        for alpha in alphas or CONFIG['alphas']:
            point=(base+direction*alpha).detach().requires_grad_(True)
            value=forward(point)
            gradient,=torch.autograd.grad(value,point)
            if not torch.isfinite(gradient).all() or not torch.isfinite(value).all():raise ValueError('nonfinite derivative')
            derivative=float((gradient.double()*direction.double()).sum());checks=[]
            for eps in epsilons or CONFIG['epsilons']:
                with torch.no_grad():
                    plus=float(forward(base+direction*(alpha+eps)));minus=float(forward(base+direction*(alpha-eps)))
                if not np.isfinite([plus,minus]).all():raise ValueError('nonfinite finite difference')
                fd=(plus-minus)/(2*eps);error=abs(derivative-fd)
                checks.append(dict(epsilon=eps,plus=plus,minus=minus,finite_difference=fd,absolute_error=error,
                    relative_error=error/max(abs(derivative),abs(fd),1e-12),
                    within_tolerance=error<=max(CONFIG['fd_atol'],CONFIG['fd_rtol']*max(abs(derivative),abs(fd)))))
            result.append(dict(alpha=alpha,logit=float(value.detach()),autograd=derivative,
                gradient_norm=float(gradient.double().norm()),finite_differences=checks))
    return result

class NativeNumerics(ig.NativeAttributor):
    def context(self,backend):
        if backend=='default':return nullcontext()
        if backend=='sdpa_math':
            from torch.nn.attention import sdpa_kernel,SDPBackend
            return sdpa_kernel(SDPBackend.MATH)
        raise ValueError('invalid attention context')

    def attention(self):
        return dict(config=getattr(self.engine.model.config,'_attn_implementation',None),
                    modules=sorted({type(m).__name__ for m in self.engine.model.modules() if 'Attention' in type(m).__name__}),
                    math_context='forces SDPA MATH only if model uses SDPA; no-op for eager')

    def inputs(self,image,expected,kind):
        x,g=self.tensor(image);b,bg=self.tensor(ig.baseline(image,kind));assert bg==g
        with self.torch.no_grad():
            feature=self.cls(x)[0].cpu().numpy()
        if not np.allclose(feature,expected,atol=1e-5,rtol=1e-4):raise ValueError('frozen target feature mismatch')
        return x,b,g,feature

    def execute(self,image,expected,kind,method,steps):
        torch=self.torch;before=Counter(self.stats)
        versions=[p._version for p in self.engine.model.parameters()]
        x,b,g,feature=self.inputs(image,expected,kind)
        target=float((((feature.astype(float)-self.head['mean'])/self.head['scale'])*self.head['coef']).sum()+self.head['intercept'])
        if method=='derivatives':
            checks={}
            for backend in CONFIG['backends']:
                with self.context(backend):
                    def forward(t):
                        if t.requires_grad:self.stats['backward_passes']+=1
                        return self.logit(t)
                    checks[backend]=directional(forward,x,b)
            result=dict(checks=checks,attention=self.attention())
        else:
            with torch.no_grad():base=float(self.logit(b))
            def forward(t):
                self.stats['backward_passes']+=1
                return self.logit(t)
            attr32,attr64=integrate_pair(forward,x,b,method,steps)
            maps=dict(fp32=attr32.sum(1)[0].cpu().numpy(),fp64=attr64.sum(1)[0].cpu().numpy())
            result=dict(raw_maps=maps,baseline_logit=base,
                diagnostics={k:completeness(v,target,base) for k,v in maps.items()},
                accumulation_comparison=similarity(maps['fp32'],maps['fp64']))
        if versions!=[p._version for p in self.engine.model.parameters()] or any(p.requires_grad or p.grad is not None for p in self.engine.model.parameters()):
            raise ValueError('model weights/gradient state changed')
        input_hash=lambda t:hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
        return dict(**result,target_feature=feature,target_logit=target,geometry=g,
                    original_input_sha256=input_hash(x),baseline_input_sha256=input_hash(b),
                    forward_passes=self.stats['forward_passes']-before['forward_passes'],
                    backward_passes=self.stats['backward_passes']-before['backward_passes'],weights_unchanged=True)
