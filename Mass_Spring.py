import taichi as ti
################################################################################
ti.init(arch = ti.cpu)
################################################################################
def main():
    gui = ti.GUI('', background_color = 0xDDDDDD)
    #
    while True:
        gui.show()

################################################################################
if __name__ == '__main__':
    main()
