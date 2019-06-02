"""Rosswog SPH
"""

from math import exp
from compyle.api import declare
from pysph.sph.equation import Equation
from pysph.sph.integrator import IntegratorStep
from pysph.base.particle_array import get_ghost_tag

GHOST_TAG = get_ghost_tag()


# properties needed
#   X: rosswog additional quantity for density, p^k, k ~ [0, 0.05], scalar
#   kX: temp variable to store denominator of V, scalar
#   V: generaized volume, scalar
#   c_mat: (dim X dim) matrix
#   aS: change in momentum per baryon/dt, (dim) vector
#   aer: change in energy per baryon/dt, scalar
#   Sx, y, z: momentum per baryon, 3 scalars for 3 dims
#   mr: baryon mass
#   er: energy per baryon 


def inverse3(mat=[0,0], imat=[0,0]):
    det, detinv = declare('double', 2)
    det = (
        mat[0] * (mat[4] * mat[8] - mat[7] * mat[5])
        - mat[1] * (mat[3] * mat[8] - mat[6] * mat[5])
        + mat[2] * (mat[3] * mat[7] - mat[6] * mat[4])
    )

    detinv = 1.0/det
    imat[0] = detinv * (mat[4]*mat[8] - mat[5]*mat[7])
    imat[1] = detinv * (mat[3]*mat[8] - mat[5]*mat[6])
    imat[2] = detinv * (mat[3]*mat[7] - mat[4]*mat[6])

    imat[3] = detinv * (mat[1]*mat[8] - mat[2]*mat[7])
    imat[4] = detinv * (mat[0]*mat[8] - mat[2]*mat[6])
    imat[5] = detinv * (mat[0]*mat[7] - mat[1]*mat[6])

    imat[6] = detinv * (mat[1]*mat[5] - mat[2]*mat[4])
    imat[7] = detinv * (mat[0]*mat[5] - mat[2]*mat[3])
    imat[8] = detinv * (mat[0]*mat[4] - mat[1]*mat[3])


def inverse2(mat=[0,0], imat=[0,0]):
    det, detinv = declare('double', 2)
    det = (
        mat[0]*mat[3] - mat[1]*mat[2]
    )

    detinv = 1.0/det
    imat[0] = detinv * mat[3]
    imat[1] = -detinv * mat[2]
    imat[2] = -detinv * mat[1]
    imat[3] = detinv * mat[0]


class VolumeWeightPressure(Equation):
    def __init__(self, dest, sources=None, k=0.05):
        super(VolumeWeightPressure, self).__init__(dest, sources)
        self.k = k

    def initialize(self, d_X, d_p, d_idx):
        k = self.k
        d_X[d_idx] = d_p[d_idx]**k


class VolumeWeightMass(Equation):
    def __init__(self, dest, sources=None, k=0.0):
        super(VolumeWeightMass, self).__init__(dest, sources)

    def initialize(self, d_X, d_mr, d_idx):
        d_X[d_idx] = d_mr[d_idx]


class SummationDensityRosswog(Equation):
    def __init__(self, dest, sources):
        super(SummationDensityRosswog, self).__init__(dest, sources)

    def initialize(self, d_idx, d_kX):
        d_kX[d_idx] = 0.

    def loop(self, d_idx, s_idx, d_X, s_X, d_kX, WI):
        d_kX[d_idx] += s_X[s_idx] * WI
    
    def post_loop(self, d_idx, d_kX, d_V, d_N, d_X, d_mr):
        d_V[d_idx] = d_X[d_idx] / d_kX[d_idx]
        d_N[d_idx] = d_mr[d_idx] / d_V[d_idx]


class KernelMatrixRosswogIA(Equation):
    def __init__(self, dest, sources, dim=2):
        super(KernelMatrixRosswogIA, self).__init__(dest, sources)
        self.dim = dim
        # self.matrix_str = 'matrix(%s)' % (self.dim * self.dim)

    def initialize(self):
        pass

    def _get_helpers_(self):
        return [inverse2, inverse3]


    def loop_all(self, d_idx, d_c_mat, d_x, d_y, d_z, d_h, s_x,
                 s_y, s_z, s_h, s_V, SPH_KERNEL, NBRS, N_NBRS):
        x = d_x[d_idx]
        y = d_y[d_idx]
        z = d_z[d_idx]
        h = d_h[d_idx]
        i, j, s_idx, d = declare('int', 4)
        d = self.dim
        xij = declare('matrix(3)')
        # tau_mat = declare(f'matrix({d})')
        tau_mat = declare('matrix(9)')
        
        for i in range(d):
            for j in range(d):
                # xj = xij[j]
                tau_mat[i * d + j] = 0. 

        for k in range(N_NBRS):
            s_idx = NBRS[k]
            Vj = s_V[s_idx]
            xij[0] = x - s_x[s_idx]
            xij[1] = y - s_y[s_idx]
            xij[2] = z - s_z[s_idx]
            r = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2])
            wij = SPH_KERNEL.kernel(xij, r, h)
            if r > 1.0e-12:
                for i in range(d):
                    for j in range(d):
                        # xj = xij[j]
                        tau_mat[i * d + j] += Vj * xij[i] * xij[j] * wij

        c_mat = declare('matrix(9)')
        if d == 1:
            c_mat[0] = 1 / tau_mat[0]
        elif d == 2:
            inverse2(tau_mat, c_mat)
        elif d == 3:
            inverse3(tau_mat, c_mat)
        
        for i in range(d * d):
            d_c_mat[9 * d_idx + i] = c_mat[i]
            # print("c_mat for ", d_idx, " is ", c_mat[i])


class AccelRosswogIA(Equation):
    def __init__(self, dest, sources, dim=2):
        super(AccelRosswogIA, self).__init__(dest, sources)
        self.dim = dim
    
    def initialize(self, d_idx, d_aS, d_aer, d_aSx, d_aSy, d_aSz):
        i = declare('int')
        for i in range(self.dim):
            d_aS[3 * d_idx + i] = 0.0
        d_aer[d_idx] = 0.0
        d_aSx[d_idx] = 0.0
        d_aSy[d_idx] = 0.0
        d_aSz[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_aS, XIJ, d_c_mat, s_c_mat,
             d_p, s_p, d_V, s_V, d_X, s_X, d_aer, d_u, d_v,
             d_w, s_u, s_v, s_w, s_orig_idx, WI, WJ, DWI, DWJ):
        # TODO: use d = 3 to fix out of order problems (x-z, no y)
        d, i = declare('int', 2)
        d_start_idx, s_start_idx = declare('int', 2)
        d = self.dim
        Ga, Gb = declare('matrix(9)', 2)
        VI, VJ = declare('matrix(3)', 2)

        VI[0] = d_u[d_idx]
        VI[1] = d_v[d_idx]
        VI[2] = d_w[d_idx]

        VJ[0] = s_u[s_idx]
        VJ[1] = s_v[s_idx]
        VJ[2] = s_w[s_idx]

        s_start_idx = 9 * s_idx
        d_start_idx = 9 * d_idx
        
        for i in range(d):
            Ga[i] = 0.0
            Gb[i] = 0.0
        # i = declare('int')
        tmp = declare('float')
        for i in range(d):
            Ga[i] = - WI * (d_c_mat[d_start_idx + i * d + 0] * XIJ[0] +\
                            d_c_mat[d_start_idx + i * d + 1] * XIJ[1] +\
                            d_c_mat[d_start_idx + i * d + 2] * XIJ[2])
            
            Gb[i] = - WJ * (s_c_mat[s_start_idx + i * d + 0] * XIJ[0] +\
                            s_c_mat[s_start_idx + i * d + 1] * XIJ[1] +\
                            s_c_mat[s_start_idx + i * d + 2] * XIJ[2])

            # print("for particle ", d_idx, " neighbour ", s_idx, " Ga ", Ga[i])
            # print("for particle ", d_idx, " neighbour ", s_idx, " Gb ", Gb[i])
            # print("for particle ", d_idx, " neighbour ", s_idx, " DWI ", DWI[i])
            # print("for particle ", d_idx, " neighbour ", s_idx, " DWJ ", DWJ[i])

            # tmp = 0.0
            d_aS[3 * d_idx + i] += d_p[d_idx] * d_V[d_idx] * d_V[d_idx] *\
                                   s_X[s_idx] * Ga[i] / d_X[d_idx] +\
                                   s_p[s_idx] * s_V[s_idx] * s_V[s_idx] *\
                                   d_X[d_idx] * Gb[i] / s_X[s_idx]

            # tmp = d_p[d_idx] * d_V[d_idx] * d_V[d_idx] *\
            #       s_X[s_idx] * Ga[i] / d_X[d_idx] +\
            #       s_p[s_idx] * s_V[s_idx] * s_V[s_idx] *\
            #       d_X[d_idx] * Gb[i] / s_X[s_idx]
            # if d_idx in [0, 3]:
            #     print("accel+ for ", d_idx, " is ", tmp, " neighbour ", s_idx, " orig id ", s_orig_idx[s_idx])
            # print('accel for ', d_idx, ' ', d_aS[3 * d_idx + i])
            # print("s_c_mat for ", d_idx, " is ", s_c_mat[s_start_idx])
            d_aer[d_idx] += d_p[d_idx] * d_V[d_idx] * d_V[d_idx] *\
                            s_X[s_idx] * Ga[i] * VJ[i] / d_X[d_idx] +\
                            s_p[s_idx] * s_V[s_idx] * s_V[s_idx] *\
                            d_X[d_idx] * Gb[i] * VI[i] / s_X[s_idx]
    
    def post_loop(self, d_idx, d_aSx, d_aSy, d_aSz, d_aer, d_mr, d_aS,
                  d_h, d_dt_cfl):
        d_aer[d_idx] = - d_aer[d_idx] / d_mr[d_idx]
        d_aSx[d_idx] = - d_aS[3 * d_idx + 0] / d_mr[d_idx]
        d_aSy[d_idx] = - d_aS[3 * d_idx + 1] / d_mr[d_idx]
        d_aSz[d_idx] = - d_aS[3 * d_idx + 2] / d_mr[d_idx]
        # print("aSx for id: ", d_idx, " is ", d_aS[3 * d_idx])

        mag = declare('float')
        mag = (
            d_aS[3 * d_idx + 0] * d_aS[3 * d_idx + 0] +
            d_aS[3 * d_idx + 1] * d_aS[3 * d_idx + 1] +
            d_aS[3 * d_idx + 2] * d_aS[3 * d_idx + 2]
        )**0.5

        d_dt_cfl[d_idx] = (mag * d_h[d_idx])**0.5



class EOSRosswog(Equation):
    def __init__(self, dest, sources=None, Gamma=5/3):
        super(EOSRosswog, self).__init__(dest, sources)
        self.Gamma = Gamma

    def initialize(self, d_idx, d_N, d_e, d_p, d_gamma):
        n = d_N[d_idx] / d_gamma[d_idx]
        d_p[d_idx] = (self.Gamma - 1) * n * d_e[d_idx]

    
class EnthalpyRosswog(Equation):
    def __init__(self, dest, sources=None):
        super(EnthalpyRosswog, self).__init__(dest, sources)

    def initialize(self, d_idx, d_eth, d_e, d_p, d_N, d_gamma):
        d_eth[d_idx] = 1 + d_e[d_idx] +\
                       d_p[d_idx] * d_gamma[d_idx] / d_N[d_idx]


class SpeedOfSoundRosswog(Equation):
    def __init__(self, dest, sources=None, Gamma=5/3):
        super(SpeedOfSoundRosswog, self).__init__(dest, sources)
        self.Gamma = Gamma

    def initialize(self, d_idx, d_cs, d_eth):
        d_cs[d_idx] = ((self.Gamma - 1) *
                       (d_eth[d_idx] - 1) / d_eth[d_idx])**0.5


class MomentumRosswog(Equation):
    def __init__(self, dest, sources=None):
        super(MomentumRosswog, self).__init__(dest, sources)

    def initialize(self, d_idx, d_Sx, d_Sy, d_Sz, d_gamma, d_er, d_eth,
                   d_u, d_v, d_w, d_e, d_p, d_N, d_rho, dt, t):
        if t < dt / 2:
            # d_N[d_idx] = d_rho[d_idx]
            # printf(" %f ", d_N[d_idx])
            d_er[d_idx] = d_gamma[d_idx] * d_eth[d_idx] -\
                          d_p[d_idx] / d_N[d_idx]
            d_Sx[d_idx] = d_gamma[d_idx] * d_u[d_idx] * (1 +
                          d_e[d_idx] + d_p[d_idx] *
                          d_gamma[d_idx] / d_N[d_idx])
            
            d_Sy[d_idx] = d_gamma[d_idx] * d_v[d_idx] * (1 +
                          d_e[d_idx] + d_p[d_idx] *
                          d_gamma[d_idx] / d_N[d_idx])
            
            d_Sz[d_idx] = d_gamma[d_idx] * d_w[d_idx] * (1 +
                          d_e[d_idx] + d_p[d_idx] *
                          d_gamma[d_idx] / d_N[d_idx])
            # printf("d_Sx for particle %d is %f\n", d_idx, d_Sx[d_idx]);
            

class RecoverPrimitiveVariables(Equation):
    def __init__(self, dest, sources=None, dim=2, tol=1e-6, Gamma=5/3):
        super(RecoverPrimitiveVariables, self).__init__(dest, sources)
        self.dim = dim
        self.tol = tol
        self.Gamma = Gamma

    def initialize(self, d_idx, d_converged, d_Sx, d_Sy, d_Sz,
                   d_p, d_N, d_er, d_gamma, d_e, d_u, d_v, d_w):
        """non relativistic implementation, assume gamma=1
        """
        # d_gamma[d_idx] = 1.0
        # printf(" %f ", d_N[d_idx])
        d_e[d_idx] = d_er[d_idx] - 1
        d_p[d_idx] = (self.Gamma - 1) * d_N[d_idx] * d_e[d_idx] /\
                     d_gamma[d_idx]
        d_u[d_idx] = d_Sx[d_idx] / (d_er[d_idx] + d_p[d_idx] / d_N[d_idx])
        d_v[d_idx] = d_Sy[d_idx] / (d_er[d_idx] + d_p[d_idx] / d_N[d_idx])
        d_w[d_idx] = d_Sz[d_idx] / (d_er[d_idx] + d_p[d_idx] / d_N[d_idx])
        # print('e for ', d_idx, ' ', d_e[d_idx])

        

class UpdateSmoothingLength(Equation):
    def __init__(self, dest, sources=None, eta=2.0, dim=2):
        super(UpdateSmoothingLength, self).__init__(dest, sources)
        self.eta = eta
        self.dim = dim

    def initialize(self, d_idx, d_h, d_V):
        d_h[d_idx] = self.eta * d_V[d_idx]**(1.0 / self.dim)


class RSPHTVDRK3Step(IntegratorStep):
    """TVD RK3 stepper for Rosswog SPH 
    """
    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_Sx0, d_Sy0, d_Sz0, d_Sx, d_Sy, d_Sz, d_er0, d_er):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_Sx0[d_idx] = d_Sx[d_idx]
        d_Sy0[d_idx] = d_Sy[d_idx]
        d_Sz0[d_idx] = d_Sz[d_idx]

        d_er0[d_idx] = d_er[d_idx]

    def stage1(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
               d_Sx0, d_Sy0, d_Sz0, d_u, d_v, d_w, d_er0, d_er,
               d_Sx, d_Sy, d_Sz, d_aSx, d_aSy, d_aSz, d_aer,
               d_N, d_gamma, d_p, d_e, d_Gamma,
               dt):
        # update momentum per baryon
        d_Sx[d_idx] = d_Sx0[d_idx] + d_aSx[d_idx] * dt
        d_Sy[d_idx] = d_Sy0[d_idx] + d_aSy[d_idx] * dt
        d_Sz[d_idx] = d_Sz0[d_idx] + d_aSz[d_idx] * dt

        # update energy per baryon
        d_er[d_idx] = d_er0[d_idx] + d_aer[d_idx] * dt

        # recover primitive variables
        d_e[d_idx] = d_er[d_idx] - 1
        d_p[d_idx] = (d_Gamma[d_idx] - 1) * d_N[d_idx] * d_e[d_idx] /\
                     d_gamma[d_idx]
        d_u[d_idx] = d_Sx[d_idx] / (d_er[d_idx] + d_p[d_idx] / d_N[d_idx])
        d_v[d_idx] = d_Sy[d_idx] / (d_er[d_idx] + d_p[d_idx] / d_N[d_idx])
        d_w[d_idx] = d_Sz[d_idx] / (d_er[d_idx] + d_p[d_idx] / d_N[d_idx])

        d_x[d_idx] = d_x0[d_idx] + d_u[d_idx] * dt
        d_y[d_idx] = d_y0[d_idx] + d_v[d_idx] * dt
        d_z[d_idx] = d_z0[d_idx] + d_w[d_idx] * dt

    def stage2(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
               d_Sx0, d_Sy0, d_Sz0, d_u, d_v, d_w, d_er0, d_er,
               d_Sx, d_Sy, d_Sz, d_aSx, d_aSy, d_aSz, d_aer,
               d_N, d_gamma, d_p, d_e, d_Gamma,
               dt):
        # update momentum per baryon
        d_Sx[d_idx] = 0.75 * d_Sx0[d_idx] + 0.25 * d_aSx[d_idx] * dt +\
                      0.25 * d_Sx[d_idx]
        d_Sy[d_idx] = 0.75 * d_Sy0[d_idx] + 0.25 * d_aSy[d_idx] * dt +\
                      0.25 * d_Sy[d_idx]
        d_Sz[d_idx] = 0.75 * d_Sz0[d_idx] + 0.25 * d_aSz[d_idx] * dt +\
                      0.25 * d_Sz[d_idx]

        # update energy per baryon
        d_er[d_idx] = 0.75 * d_er0[d_idx] + 0.25 * d_aer[d_idx] * dt +\
                      0.25 * d_er[d_idx]

        # recover primitive variables
        d_e[d_idx] = d_er[d_idx] - 1
        d_p[d_idx] = (d_Gamma[d_idx] - 1) * d_N[d_idx] * d_e[d_idx] /\
                     d_gamma[d_idx]
        d_u[d_idx] = d_Sx[d_idx] / (d_er[d_idx] + d_p[d_idx] / d_N[d_idx])
        d_v[d_idx] = d_Sy[d_idx] / (d_er[d_idx] + d_p[d_idx] / d_N[d_idx])
        d_w[d_idx] = d_Sz[d_idx] / (d_er[d_idx] + d_p[d_idx] / d_N[d_idx])

        d_x[d_idx] = 0.75 * d_x0[d_idx] + 0.25 * d_u[d_idx] * dt +\
                     0.25 * d_x[d_idx]
        d_y[d_idx] = 0.75 * d_y0[d_idx] + 0.25 * d_v[d_idx] * dt +\
                     0.25 * d_y[d_idx]
        d_z[d_idx] = 0.75 * d_z0[d_idx] + 0.25 * d_w[d_idx] * dt +\
                     0.25 * d_z[d_idx]

    def stage3(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
               d_Sx0, d_Sy0, d_Sz0, d_u, d_v, d_w, d_er0, d_er,
               d_Sx, d_Sy, d_Sz, d_aSx, d_aSy, d_aSz, d_aer,
               d_N, d_gamma, d_p, d_e, d_Gamma,
               dt):
        oneby3 = 1./3.
        twoby3 = 2./3.
        # update momentum per baryon
        d_Sx[d_idx] = oneby3 * d_Sx0[d_idx] + twoby3 * d_aSx[d_idx] * dt +\
                      twoby3 * d_Sx[d_idx]
        d_Sy[d_idx] = oneby3 * d_Sy0[d_idx] + twoby3 * d_aSy[d_idx] * dt +\
                      twoby3 * d_Sy[d_idx]
        d_Sz[d_idx] = oneby3 * d_Sz0[d_idx] + twoby3 * d_aSz[d_idx] * dt +\
                      twoby3 * d_Sz[d_idx]

        # update energy per baryon
        d_er[d_idx] = oneby3 * d_er0[d_idx] + twoby3 * d_aer[d_idx] * dt +\
                      twoby3 * d_er[d_idx]

        # recover primitive variables
        d_e[d_idx] = d_er[d_idx] - 1
        d_p[d_idx] = (d_Gamma[d_idx] - 1) * d_N[d_idx] * d_e[d_idx] /\
                     d_gamma[d_idx]
        d_u[d_idx] = d_Sx[d_idx] / (d_er[d_idx] + d_p[d_idx] / d_N[d_idx])
        d_v[d_idx] = d_Sy[d_idx] / (d_er[d_idx] + d_p[d_idx] / d_N[d_idx])
        d_w[d_idx] = d_Sz[d_idx] / (d_er[d_idx] + d_p[d_idx] / d_N[d_idx])

        d_x[d_idx] = oneby3 * d_x0[d_idx] + twoby3 * d_u[d_idx] * dt +\
                     twoby3 * d_x[d_idx]
        d_y[d_idx] = oneby3 * d_y0[d_idx] + twoby3 * d_v[d_idx] * dt +\
                     twoby3 * d_y[d_idx]
        d_z[d_idx] = oneby3 * d_z0[d_idx] + twoby3 * d_w[d_idx] * dt +\
                     twoby3 * d_z[d_idx]


class RSPHEulerStep(IntegratorStep):
    def stage1(self, d_idx, d_Sx, d_Sy, d_Sz, d_aSx, d_aSy, d_aSz,
               d_e, d_er, d_aer, d_x, d_y, d_z, d_u, d_v, d_w,
               d_Gamma, d_N, d_p, d_gamma, dt):
        d_Sx[d_idx] += d_aSx[d_idx] * dt
        d_Sy[d_idx] += d_aSy[d_idx] * dt
        d_Sz[d_idx] += d_aSz[d_idx] * dt

        # print("dt, ", dt)
        # update energy per baryon
        d_er[d_idx] += d_aer[d_idx] * dt

        u0 = d_u[d_idx]
        v0 = d_v[d_idx]
        w0 = d_w[d_idx]

        # recover primitive variables
        d_e[d_idx] = d_er[d_idx] - 1
        d_p[d_idx] = (d_Gamma[d_idx] - 1) * d_N[d_idx] * d_e[d_idx] /\
                     d_gamma[d_idx]
        d_u[d_idx] = d_Sx[d_idx] / (d_er[d_idx] + d_p[d_idx] / d_N[d_idx])
        d_v[d_idx] = d_Sy[d_idx] / (d_er[d_idx] + d_p[d_idx] / d_N[d_idx])
        d_w[d_idx] = d_Sz[d_idx] / (d_er[d_idx] + d_p[d_idx] / d_N[d_idx])

        d_x[d_idx] += (u0 + d_u[d_idx]) * dt * 0.5
        d_y[d_idx] += (v0 + d_v[d_idx]) * dt * 0.5
        d_z[d_idx] += (w0 + d_w[d_idx]) * dt * 0.5
        # printf("velocity diff for %d is %.9g\n", d_idx,  d_u[d_idx] - u0)


class DebugEqn(Equation):
    def __init__(self, dest, sources=None):
        super(DebugEqn, self).__init__(dest, sources)

    def initialize(self, d_idx, d_h, d_c_mat, d_p, d_e, d_N, d_V, d_x, d_X):
        return 0
        print("p for ", d_idx, " ", d_p[d_idx])
        print("h for ", d_idx, " ", d_h[d_idx])
        print("X for ", d_idx, " ", d_X[d_idx])
        print("c_mat for ", d_idx, " ", d_c_mat[9 * d_idx])
        print("V for ", d_idx, " ", d_V[d_idx])
        print("N for ", d_idx, " ", d_N[d_idx])
        print("e for ", d_idx, " ", d_e[d_idx])
        print("x for ", d_idx, " ", d_x[d_idx])


class RSPHUpdateGhostProps(Equation):
    """Copy the RSPH gradients and other props required for RSPH
    from real particle to ghost particles

    """
    def __init__(self, dest, sources=None):
        super(RSPHUpdateGhostProps, self).__init__(dest, sources)
        assert GHOST_TAG == 2

    def initialize(self, d_idx, d_tag, d_orig_idx, d_c_mat,
                   d_cs, d_eth, d_X, d_V, d_N, d_p):
        idx, i = declare('int', 2)
        if d_tag[d_idx] == 2:
            idx = d_orig_idx[d_idx]
            for i in range(9):
                d_c_mat[d_idx * 9 + i] = d_c_mat[idx * 9 + i]
            d_cs[d_idx] = d_cs[idx]
            d_N[d_idx] = d_N[idx]
            d_eth[d_idx] = d_eth[idx]
            d_X[d_idx] = d_X[idx]
            d_V[d_idx] = d_V[idx]
            d_p[d_idx] = d_p[idx]
