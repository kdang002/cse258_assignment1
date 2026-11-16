import implicit.gpu as g

print("implicit_cuda", bool(g.HAS_CUDA))
print("implicit_rmm", bool(g.HAS_RMM))
if g.HAS_CUDA:
    print("gpu_count", g.get_device_count())
