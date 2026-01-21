import numpy as np

class FiniteElement1D:
    def __init__(self, f):
        self.f = f

    # Função para obtenção das bases e dos pesos
    def shg_w(self, nint, nen):
        shg1 = np.zeros((nen, nint)) # Funções de base
        shg2 = np.zeros((nen, nint)) # Derivadas das funções de base
        if nint == 2:
            pt = np.array([-np.sqrt(3)/3, np.sqrt(3)/3])
            w = np.array([1.0, 1.0])
        elif nint == 3:
            pt = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
            w = np.array([5/9, 8/9, 5/9])
        elif nint == 4:
            pt = np.array([-np.sqrt(3/7 + (2/7)*np.sqrt(6/5)), -np.sqrt(3/7 - (2/7)*np.sqrt(6/5)),
                        np.sqrt(3/7 - (2/7)*np.sqrt(6/5)),
                        np.sqrt(3/7 + (2/7)*np.sqrt(6/5))])
            w = np.array([(18 - np.sqrt(30))/36, (18 + np.sqrt(30))/36, (18 + np.sqrt(30))/36, (18 - np.sqrt(30))/36])
        elif nint == 5:
            pt = np.array([-(1/3)*np.sqrt(5 + 2*np.sqrt(10/7)), -(1/3)*np.sqrt(5 - 2*np.sqrt(10/7)), 0,
                        (1/3)*np.sqrt(5 - 2*np.sqrt(10/7)), (1/3)*np.sqrt(5 + 2*np.sqrt(10/7))])
            w = np.array([(322 - 13*np.sqrt(70))/900, (322 + 13*np.sqrt(70))/900, 128/225,
                        (322 + 13*np.sqrt(70))/900, (322 - 13*np.sqrt(70))/900])
            
        for l in range(nint):
            t = pt[l]
            if nen == 2:
                shg1[0, l] = 0.5 * (1 - t)
                shg1[1, l] = 0.5 * (1 + t)
                # Derivadas para nen = 2
                shg2[0, l] = -0.5
                shg2[1, l] = 0.5
            elif nen == 3:
                shg1[0, l] = 0.5 * t * (t - 1)
                shg1[1, l] = - (t - 1) * (t + 1)
                shg1[2, l] = 0.5 * t * (t + 1)
                # Derivadas para nen = 3
                shg2[0, l] = t - 0.5
                shg2[1, l] = -2 * t
                shg2[2, l] = t + 0.5
            elif nen == 4:
                shg1[0, l] = -9/16*(t + 1/3)*(t - 1/3)*(t - 1)
                shg1[1, l] = 27/16*(t + 1)*(t - 1/3)*(t - 1)
                shg1[2, l] = -27/16*(t + 1)*(t + 1/3)*(t - 1)
                shg1[3, l] = 9/16*(t + 1)*(t + 1/3)*(t - 1/3)
                # Derivadas para nen = 4
                shg2[0, l] = -27/16 * t**2 + 9/8 * t + 1/16
                shg2[1, l] = 81/16 * t**2 - 9/8 * t - 27/16
                shg2[2, l] = -81/16 * t**2 - 9/8 * t + 27/16
                shg2[3, l] = 27/16 * t**2 + 9/8 * t - 1/16
            elif nen == 5:
                shg1[0, l] = (2/3)*(t + 1/2)*t*(t - 1/2)*(t - 1)
                shg1[1, l] = -(8/3)*(t + 1)*t*(t - 1/2)*(t - 1)
                shg1[2, l] = 4*(t + 1)*(t + 1/2)*(t - 1/2)*(t - 1)
                shg1[3, l] = - (8/3)*(t + 1)*(t + 1/2)*t*(t - 1)
                shg1[4, l] = (2/3)*(t + 1)*(t + 1/2)*t*(t - 1/2)
                # Derivadas para nen = 5
                shg2[0, l] = 8/3 * t**3 - 2 * t**2 - 1/3 * t + 1/6
                shg2[1, l] = -32/3 * t**3 + 4 * t**2 + 16/3 * t - 4/3
                shg2[2, l] = 16 * t**3 - 10 * t
                shg2[3, l] = -32/3 * t**3 - 4 * t**2 + 16/3 * t + 4/3
                shg2[4, l] = 8/3 * t**3 + 2 * t**2 - 1/3 * t - 1/6

        return shg1, shg2, w
    
    # Construção das matrizes K e F
    def make_matrices(self, a, b, nel, nint, nen, u_a, u_b):
        # Parâmetros
        h = (b - a) / nel
        # Funções base e pesos
        shg1, shg2, w = self.shg_w(nint, nen)
        tam = (nen-1)*nel+1
        x_global = np.linspace(a, b, tam)

        K = np.zeros((tam, tam))
        F = np.zeros(tam)
        
        # Algoritmo apresentado em aula para construção das matrizes locais e globais
        for n in range(nel):
            Ke = np.zeros((nen, nen))
            Fe = np.zeros(nen)
            idx_start, idx_end = n*(nen - 1), n*(nen - 1) + nen
            indices = list(range(idx_start, idx_end))
            xl = x_global[idx_start:idx_end]
            for l in range(nint):
                xx = 0
                for i in range(nen):
                    xx += shg1[i, l]*xl[i]
                for j in range(nen):
                    Fe[j] += self.f(xx)*shg1[j, l]*w[l]*(h/2)
                    for i in range(nen):
                        Ke[i, j] += shg2[i, l]*(2/h)*shg2[j, l]*(2/h)*w[l]*(h/2)
        
            for i in range(nen):
                F[indices[i]] += Fe[i]
                for j in range(nen):
                    K[indices[i], indices[j]] += Ke[i, j]

        # right Dirichlet boundary condition
        K[0, :] = 0
        K[0, 0] = 1
        F[0] = u_a

        # left Dirichlet boundary condition
        K[-1, :] = 0
        K[-1, -1] = 1
        F[-1] = u_b
        
        return K, F

    def erro_L2(self, u, u_exata, a, b, nel, nint, nen):
        erul2 = 0
        h = (b - a)/nel
        shg1, _, w = self.shg_w(nint, nen)
        tam = (nen-1)*nel+1
        x_global = np.linspace(a, b, tam)
        
        for n in range(nel):
            idx_start, idx_end = n*(nen - 1), n*(nen - 1) + nen
            indices = list(range(idx_start, idx_end))
            xl = x_global[idx_start:idx_end]
            eru = 0
            for l in range(nint):
                uh = 0
                xx = 0
                for i in range(nen):
                    uh += shg1[i, l]*u[indices[i]]
                    xx += shg1[i, l]*xl[i]
                eru += ((u_exata(xx) - uh)**2)*w[l]*(h/2)
            erul2 += eru
        
        return np.sqrt(erul2)
