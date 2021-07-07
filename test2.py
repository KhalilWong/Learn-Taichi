import taichi as ti
#
ti.init(arch = ti.gpu)
#
blk1 = ti.root
blk2 = blk1.dense(ti.i,  3)
blk3 = blk2.dense(ti.jk, (5, 2))
blk4 = blk3.dense(ti.k,  2)
blk1.shape()  # ()
blk2.shape()  # (3, )
blk3.shape()  # (3, 5, 2)
blk4.shape()  # (3, 5, 4)
#
@ti.kernel
def my_kernel():
    for i in x:
        x[i] = i * 2

x = ti.field(ti.f32, 4)
my_kernel()
x_np = x.to_numpy()
print(x_np)  # np.array([0, 2, 4, 6])
################################################################################
