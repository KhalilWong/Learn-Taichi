import taichi as ti
#
ti.init(arch = ti.gpu)
#
@ti.kernel
def my_kernel():
    print(x[0])
    for i in x:
        print(i)
        x[i] = i * 2

x = ti.field(ti.f32, 4)
my_kernel()
x_np = x.to_numpy()
print(x_np)  # np.array([0, 2, 4, 6])
