"""Simulate a shocktube reflect problem in 1D (15 seconds).
"""
from shocktube_setup import ShockTubeSetup
from pysph.sph.scheme import (
    GasDScheme, GSPHScheme, SchemeChooser, ADKEScheme
)
from pysph.sph.scheme import ADKESchemeBoundary
import numpy

# Numerical constants
dim = 1
gamma = 1.4
gamma1 = gamma - 1.0

# solution parameters
dt = 3e-4
tf = 1.65


class SodShockTube(ShockTubeSetup):

    def initialize(self):
        self.xmin = -0.6
        self.xmax = 0.6
        self.x0 = 0.0
        self.rhol = 5.8
        self.rhor = 1.5
        self.pl = 0.5
        self.pr = 0.125
        self.ul = 0.0
        self.ur = 0.0
        self.rho0 = numpy.max((self.rhol, self.rhor))
        self.c0 = 2000
        self.p0 = numpy.max(
            (self.pl, self.pr, self.rho0*self.c0**2)
        )

    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float,
            dest="hdx", default=1.2,
            help="Ratio h/dx."
        )
        group.add_argument(
            "--nl", action="store", type=float, dest="nl", default=320,
            help="Number of particles in left region"
        )

    def consume_user_options(self):
        self.nl = self.options.nl
        self.hdx = self.options.hdx
        ratio = self.rhor/self.rhol
        self.nr = ratio*self.nl
        self.dxl = 0.6/self.nl
        self.dxr = 0.6/self.nr
        self.h0 = self.hdx * self.dxr
        self.hdx = self.hdx

    def create_particles(self):
        pa =  self.generate_particles(xmin=self.xmin, xmax=self.xmax,
                                        dxl=self.dxl, dxr=self.dxr,
                                        m=self.dxl*self.rhol, pl=self.pl,
                                        pr=self.pr, h0=self.h0, bx=0.03,
                                        gamma1=gamma1, ul=self.ul, ur=self.ur
        )

        return pa

    def create_scheme(self):
        self.dt = dt
        self.tf = tf

        adke = ADKESchemeBoundary(
            fluids=['fluid'], solids=['boundary'], dim=dim, rho0=self.rho0,
            p0=self.p0, gamma=gamma,alpha=4.0, beta=1.0, k=0.3,eps=1.0,
            g1=0.2, g2=0.4
        )

        mpm = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=1.2, alpha1=1.0, alpha2=0.1,
            beta=1.0, update_alpha1=True, update_alpha2=True
        )

        gsph = GSPHScheme(
            fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma,
            kernel_factor=1.2,
            g1=0.2, g2=0.4, rsolver=2, interpolation=1, monotonicity=1,
            interface_zero=True, hybrid=False, blend_alpha=2.0,
            niter=20, tol=1e-6
        )

        s = SchemeChooser(default='adke', adke=adke, mpm=mpm, gsph=gsph)
        return s


if __name__ == '__main__':
    app = SodShockTube()
    app.run()
    app.post_process()
