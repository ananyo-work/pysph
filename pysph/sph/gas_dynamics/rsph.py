"""Rosswog SPH
"""

from math import exp
from compyle.api import declare
from pysph.sph.equation import Equation


# properties needed
#   X: rosswog additional quantity for density, p^k, k ~ [0, 0.05], scalar
#   kX: temp variable to store denominator of V, scalar
#   V: generaized volume, scalar
#   c_mat: (dim X dim) matrix
#   aS: change in momentum per baryon/dt, (dim) vector
#   aer: change in energy per baryon/dt, scalar


def inverse3(mat, imat):
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


def inverse2(mat, imat):
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
    def __init__(self, dest, sources=None):
        super(VolumeWeightMass, self).__init__(dest, sources)

    def initialize(self, d_X, d_nu, d_idx):
        d_X[d_idx] = d_nu[d_idx]


class SummationDensityRosswog(Equation):
    def __init__(self, dest, sources):
        super(SummationDensityRosswog, self).__init__(dest, sources)

    def initialize(self, d_idx, d_kX):
        d_kX[d_idx] = 0.

    def loop(self, d_idx, s_idx, d_X, s_X, d_kX, WI):
        d_kX[d_idx] += s_X[s_idx] * WI
    
    def post_loop(self, d_idx, d_kX, d_V, d_N, d_X, d_nu):
        d_V[d_idx] = d_X[d_idx] / d_kX[d_idx]
        d_N[d_idx] = d_nu[d_idx] / d_V[d_idx]


class KernelMatrixRosswogIA(Equation):
    def __init__(self, dest, sources, dim=2):
        super(KernelMatrixRosswog, self).__init__(dest, sources)
        self.dim = dim

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
        tau_mat = declare('matrix(%s)' % (d * d))
        
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

        c_mat = declare('matrix(%s)' % (d * d))
        if d == 1:
            c_mat[0] = 1 / tau_mat[0]
        elif d == 2:
            inverse2(tau_mat, c_mat)
        elif d == 3:
            inverse3(tau_mat, c_mat)
        
        for i in range(d * d):
            d_c_mat[d * d * d_idx + i] = c_mat[i]


class AccelRosswogIA(Equation):
    def __init__(self, dest, sources, dim=2):
        super(MomentumRosswogIA, self).__init__(dest, sources)
        self.dim = dim
    
    def initialize(self, d_idx, d_aS, a_aer):
        for i in range(self.dim):
            d_aS[self.dim * d_idx + i] = 0.0
            d_aer[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_aS, XIJ, d_c_mat, s_c_mat,
             d_p, s_p, d_V, s_V, d_X, s_X, d_aer, d_u, d_v,
             d_w, s_u, s_v, s_w, WI, WJ):
        d = self.dim
        Ga, Gb = declare('matrix(%s)' % (self.dim), 2)
        VI, VJ = declare('matrix(3)', 2)

        VI[0] = d_u
        VI[1] = d_v
        VI[2] = d_w

        VJ[0] = s_u
        VJ[1] = s_v
        VJ[2] = s_w

        s_start_idx = d * d * s_idx
        
        # for i in range(d):
        #     Ga[i] = 0.0
        #     Gb[i] = 0.0
        
        for i in range(d):
            Ga[i] = WI * (d_c_mat[s_start_idx + i * d + 0] * XIJ[0] -\
                          d_c_mat[s_start_idx + i * d + 1] * XIJ[1] -\
                          d_c_mat[s_start_idx + i * d + 2] * XIJ[2])
            
            Gb[i] = WJ * (s_c_mat[s_start_idx + i * d + 0] * XIJ[0] -\
                          s_c_mat[s_start_idx + i * d + 1] * XIJ[1] -\
                          s_c_mat[s_start_idx + i * d + 2] * XIJ[2])
        
            d_aS[d * d_idx + i] += d_p[d_idx] * d_V[d_idx] * d_V[d_idx] *\
                                   s_X[s_idx] * Ga[i] / d_X[d_idx] +\
                                   s_p[s_idx] * s_V[s_idx] * s_V[s_idx] *\
                                   d_X[d_idx] * Gb[i] / s_X[s_idx]

            d_aer[d_idx] += d_p[d_idx] * d_V[d_idx] * d_V[d_idx] *\
                            s_X[s_idx] * Ga[i] * VI[i] / d_X[d_idx] +\
                            s_p[s_idx] * s_V[s_idx] * s_V[s_idx] *\
                            d_X[d_idx] * Gb[i] * VJ[i] / s_X[s_idx]
    
    def post_loop(self, d_idx, d_aSx, d_aSy, d_aSz, d_aer, d_nu):
        d = self.dim
        d_aer[d_idx] = - d_aer[d_idx] / d_nu[d_idx]
        d_aSx[d_idx] = - d_aS[d * d_idx + 0] / d_nu[d_idx]
        d_aSy[d_idx] = - d_aS[d * d_idx + 1] / d_nu[d_idx]
        d_aSz[d_idx] = - d_aS[d * d_idx + 2] / d_nu[d_idx]


class EOSRosswog(Equation):
    def __init__(self, dest, sources=None, Gamma=5/3):
        super(EOSRosswog, self).__init__(dest, sources)
        self.Gamma = Gamma

    def initialize(self, d_idx, d_N, d_e, d_p, d_gamma):
        n = d_N[d_idx] / d_gamma[d_idx]
        d_p[d_idx] = (self.Gamma - 1) * n * d_e[d_idx]