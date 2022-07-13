import numpy as np

def sensorfaultmodels(y0, T1, T2, ftype, fpar):

        # T1 = self.fault_sens_start[index_id]
        # T2 = self.fault_sens_end[index_id]
        # a1 = self.fault_a1[index_id]
        # a2 = self.fault_a2[index_id]
        # ftype = self.fault_type[index_id]
        # fpar = float(self.fault_functionpar[index_id])
        if ftype == 'constant':
            a1 = 0.5  # parameter in occurance evolution profile function
            a2 = 0.7  # parameter in dissapearance evolution profile function
        if ftype == 'drift':
            a1 = 999999
            a2 = 1
        if ftype == 'normal':
            a1 = 100
            a2 = 100
        if ftype == 'percentage':
            a1 = 999999
            a2 = .7
        if ftype == 'stuckzero':
            a1 = 999999
            a2 = .7
        
        y = []
        for k in range(0, len(y0)):
            y0k = y0[k]
            b1 = 0
            b2 = 0
            if k >= T1:
                b1 = 1 - np.exp(- a1 * (k - T1))

            if k >= T2:
                b2 = 1 - np.exp(- a2 * (k - T2))
                
            b = b1 - b2

            phi = 0

            if b > 0:
                if ftype == 'constant':
                    phi = fpar
                if ftype == 'drift':
                    phi = fpar * (k - T1)
                if ftype == 'normal':
                    phi = np.random.normal(0, fpar)
                if ftype == 'percentage':
                    phi = fpar * y0k
                if ftype == 'stuckzero':
                    phi = -y0k

            df = b * phi
            y0k = y0k + df
            y.append(y0k)
        y = np.array(y)
        return y