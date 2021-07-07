import taichi as ti
import math

################################################################################
ti.init(arch = ti.gpu)
#
@ti.data_oriented
class StarSystem():
    def __init__(self, sn, pn, dt):
        self.sn = sn
        self.pn = pn
        self.dt = dt
        self.x = ti.Vector.field(2, dtype = ti.f32, shape = pn)
        self.v = ti.Vector.field(2, dtype = ti.f32, shape = pn)
        self.centers = ti.Vector.field(2, dtype = ti.f32, shape = sn)
    #
    @staticmethod
    @ti.func
    def random_vector(radius):
        theta = ti.random() * 2 * math.pi
        r = ti.random() * radius
        return r * ti.Vector([ti.cos(theta), ti.sin(theta)])
    #
    @ti.kernel
    def initialize_particles(self):
        for i in range(self.pn):
            print(i)
            self.x[i] = [0.5, 0.5] + self.random_vector(0.5)
            self.v[i] = [0.0, 0.0]
            for j in range(self.sn):
                offset = -(self.centers[j] - self.x[i])
                self.v[i] += [-offset.y / offset.norm() ** 1.5, offset.x / offset.norm() ** 1.5]
                #self.v[i] *= 1 / offset.norm() ** 1.5   # 开普勒第三定律近似
    #
    @ti.func
    def gravity(self, pos, center):
        offset = -(pos - center)
        return offset / offset.norm() ** 3
    #
    @ti.kernel
    def integrate(self):
        for i in range(self.pn):
            for j in range(self.sn):
                self.v[i] += self.dt * self.gravity(self.x[i], self.centers[j])
            self.x[i] += self.dt * self.v[i]
    #
    def render(self, gui):
        gui.circle([0.5, 0.35], radius = 10, color = 0xffaa88)
        gui.circle([0.4, 0.65], radius = 10, color = 0xffaa88)
        gui.circle([0.6, 0.65], radius = 10, color = 0xffaa88)
        gui.circles(self.x.to_numpy(), radius = 3, color = 0xffffff)
################################################################################
solar = StarSystem(3, 8, 0.0001)
solar.centers[0] = [0.5, 0.35]
solar.centers[1] = [0.4, 0.65]
solar.centers[2] = [0.6, 0.65]
solar.initialize_particles()

gui = ti.GUI('Solar System', (1280, 720), background_color = 0x0071a)
while gui.running:
    for e in gui.get_events():
        if e.key == ti.GUI.ESCAPE:
            exit()
        elif e.key == ti.GUI.SPACE:
            solar.initialize_particles()
    #if gui.get_event() and gui.is_pressed(gui.SPACE):
    #    solar.initialize_particles()
    for i in range(10):
        solar.integrate()
    solar.render(gui)
    gui.show()
