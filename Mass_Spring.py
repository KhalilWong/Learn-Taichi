import taichi as ti

################################################################################
ti.init(arch = ti.gpu)
################################################################################
max_num_particles = 1024
dt = 1e-3
substeps = 10

x = ti.Vector.field(2, dtype = ti.f32, shape = max_num_particles)
v = ti.Vector.field(2, dtype = ti.f32, shape = max_num_particles)
f = ti.Vector.field(2, dtype = ti.f32, shape = max_num_particles)

fixed = ti.field(dtype = ti.i32, shape = max_num_particles)
num_particles = ti.field(dtype = ti.i32, shape = ())
spring_Y = ti.field(dtype = ti.f32, shape = ())
rest_length = ti.field(dtype = ti.f32, shape = (max_num_particles, max_num_particles))

################################################################################
@ti.kernel
def substep():
    n = num_particles[None]
    for i in range(n):
        f[i] = ti.Vector([0.0, -9.8])
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                d = x_ij.normalized()
                f[i] += -spring_Y[None] * (x_ij.norm() / rest_length[i, j] - 1) * d
    #
    for i in range(n):
        if not fixed[i]:
            v[i] += dt * f[i]
            x[i] += dt * v[i]
        else:
            v[i] = ti.Vector([0., 0.])
        #
        for d in ti.static(range(2)):
            old_x = x[i][d]
            x[i][d] = min(1, max(0, x[i][d]))
            if x[i][d] != old_x:
                v[i][d] = -0.7 * v[i][d]

################################################################################
@ti.kernel
def add_particle(pos_x: ti.f32, pos_y: ti.f32, fixed_: ti.i32):
    new_particle_id = num_particles[None]
    num_particles[None] += 1
    #
    x[new_particle_id] = ti.Vector([pos_x, pos_y])
    fixed[new_particle_id] = fixed_

    for i in range(num_particles[None] - 1):    # -1排除自身
        if (x[new_particle_id] - x[i]).norm() < 0.15:
            rest_length[new_particle_id, i] = 0.1
            rest_length[i, new_particle_id] = 0.1

################################################################################
def main():
    spring_Y[None] = 1000
    #
    gui = ti.GUI('Mass Spring', background_color = 0xDDDDDD)
    #
    while True:
        for i in range(substeps):
            substep()
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.LMB:
                add_particle(e.pos[0], e.pos[1], False)
            elif e.key == ti.GUI.RMB:
                add_particle(e.pos[0], e.pos[1], True)
        #
        X = x.to_numpy()
        #
        for i in range(num_particles[None]):
            for j in range(num_particles[None]):
                if rest_length[i, j] != 0:
                    gui.line(begin = X[i], end = X[j], color = 0x888888, radius = 2)
        #
        for i in range(num_particles[None]):
            c = 0xFF0000 if fixed[i] else 0x0
            gui.circle(X[i], color = c, radius = 5)
        #
        gui.show()

################################################################################
if __name__ == '__main__':
    main()
