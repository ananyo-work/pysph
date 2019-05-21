import numpy
from pysph.base.kernels import Gaussian
from pysph.base.utils import get_particle_array as gpa
from pysph.solver.application import Application
from pysph.sph.equation import Group
from pysph.solver.solver import Solver
from pysph.sph.integrator import EulerIntegrator
from pysph.sph.integrator_step import EulerStep

from pysph.sph.gas_dynamics.rsph import \
    VolumeWeightMass, VolumeWeightPressure, EOSRosswog,\
    SummationDensityRosswog

class DiscontTest(Application):
    def configure_method(self, method=0):
        self.method = method

    def create_particles(self):
        rhol = 1
        rhor = 2
        xmin = -0.25
        xmax = 0.25
        gamma = 5/3
        kernel_factor = 2.5
        p = 1
        self.nparticlesx = 100
        dx = (xmax - xmin) / self.nparticlesx
        dxb2 = dx / 2        
        self.nparticlesy = 50
        dy = dx / numpy.sqrt(12)  # regular rhexagonal lattice

        s = dx / numpy.sqrt(3)  # side of an hexagon
        v = 0.166 * 3 * numpy.sqrt(3) * s**2  # volume of hexagon
        h = kernel_factor * numpy.sqrt(v)

        x = numpy.zeros((0))
        y = numpy.zeros((0))

        for i in range(self.nparticlesy):
            tmpx = numpy.arange(
                (i%2)*dxb2 + xmin, xmax, dx
            )

            tmpy = numpy.ones_like(tmpx) * i * dy

            x = numpy.concatenate((x, tmpx))
            y = numpy.concatenate((y, tmpy))

        left_indices = numpy.where(x <= 0)[0]
        rho = numpy.ones_like(x) * rhor
        rho[left_indices] = 1
        p = numpy.ones_like(x) * p
        e = p / ((gamma - 1) * rho)
        m = rho * v
        
        fluid = gpa(
            name='fluid', x=x, y=y, V=v, p=p, u=0, v=0, nu=m, N=rho, e=e,
            gamma=1, h=h, additional_props=['X', 'kX', 'arho']
        )

        fluid.add_output_arrays(['nu', 'e', 'p', 'N'])
        return [fluid]

    def create_solver(self):
        dt = 1e-3
        tf = 1e-3
        kernel = Gaussian(dim=2)
        integrator = EulerIntegrator(fluid=EulerStep())
        solver = Solver(
            kernel=kernel, dim=2, integrator=integrator,
            dt=dt, tf=tf, adaptive_timestep=False, pfreq=1
        )

        return solver

    def create_equations(self):
        equations = []
        
        g1 = []
        if self.method == 0:
            g1.append(
                VolumeWeightPressure(dest='fluid', sources=['fluid'])
            )
        elif self.method == 1:
            g1.append(
                VolumeWeightMass(dest='fluid', sources=['fluid'])
            )
        equations.append(Group(equations=g1))
        
        g2 = []
        g2.append(
            SummationDensityRosswog(dest='fluid', sources=['fluid'])
        )
        equations.append(Group(equations=g2))

        g3 = []
        g3.append(
            EOSRosswog(dest='fluid', sources=['fluid'])
        )
        equations.append(Group(equations=g3))

        return equations

    def post_process(self):
        from pysph.solver.utils import load
        if len(self.output_files) < 1:
            return
        outfile = self.output_files[-1]
        data = load(outfile)
        pa = data['arrays']['fluid']
        x = pa.x
        y = pa.y
        rho = pa.N
        e = pa.e
        p = pa.p

        from matplotlib import pyplot
        offset = int(self.nparticlesy / 2) * self.nparticlesx
        pyplot.scatter(
            x[offset:offset + self.nparticlesx],
            e[offset:offset + self.nparticlesx],
            s=4
        )
        pyplot.scatter(
            x[offset:offset + self.nparticlesx],
            p[offset:offset + self.nparticlesx],
            s=4
        )
        pyplot.scatter(
            x[offset:offset + self.nparticlesx],
            rho[offset:offset + self.nparticlesx],
            s=4
        )
        pyplot.legend(['ie', 'pressure', 'density'])
        pyplot.xlim((-0.1, 0.1))
        # pyplot.gca().set_aspect('equal', adjustable='box')
        pyplot.show()


if __name__ == "__main__":
    app = DiscontTest()
    app.configure_method(0)
    app.run()
    app.post_process()