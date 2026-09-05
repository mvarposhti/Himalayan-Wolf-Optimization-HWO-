import numpy as np
import math
import random
def levy1(d):
    beta = 1.5
    sigma = (math.gamma(1+beta) * math.sin(math.pi*beta/2) /
             (math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)

    u = np.random.normal(0, sigma, d)
    v = np.random.normal(0, 1, d)

    step = u / (np.abs(v)**(1/beta))
    return step




def hwo(obj_func, dim, lb, ub, max_iter, pop_size=30):
    
    lb = np.array([lb] * dim) if isinstance(lb, (int, float)) else np.array(lb)
    ub = np.array([ub] * dim) if isinstance(ub, (int, float)) else np.array(ub)
    positions = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([obj_func(p) for p in positions])
    sorted_idx = np.argsort(fitness)
    gbest = positions[sorted_idx[0],:].copy()
    fit_gbest = fitness[sorted_idx[0]]
    lbest = positions[sorted_idx[0],:].copy()
    fit_lbest = fitness[sorted_idx[0]]
    pbest = positions.copy()
    fit_pbest = fitness
    Curve = np.zeros([max_iter, 1])


    for t in range(max_iter):
        t3 = -(np.cos((math.pi/2)*(t/max_iter))**2)+np.cos((math.pi/2)*(t/max_iter))
        a = 2 - t * (2.0 / max_iter)
        for i in range(pop_size):
            
            r = 0.01
            current_X = positions[i, :]
            F = a * (2 * np.random.random()-1) + t3 * np.random.normal(0,1)
            r1 = np.random.random(dim)
            A1 = 2 * a * r1 - a

            if abs(F) >= 1 and np.random.random()<0.2:
                X1 = (gbest+F+np.random.random(dim)*((ub-lb)*np.random.random(dim)+lb));
                X2 = (lbest+F+np.random.random(dim)*((ub-lb)*np.random.random(dim)+lb));
                X3 = (pbest[i,:]+F+np.random.random(dim)*((ub-lb)*np.random.random(dim)+lb));
            elif abs(F) >=1:

                X1 = gbest - abs(gbest - current_X*r*np.random.random(dim)*levy1(dim)) * F

                X2 = lbest - abs(lbest - current_X*r*np.random.random(dim)*levy1(dim)) * F

                X3 = pbest[i,:] - abs(pbest[i,:] - current_X*r*np.random.random(dim)*levy1(dim)) * F


            else:
                r1 = np.random.random(dim)
                A1 = 2 * a * r1 - a
                D_alpha = abs(gbest - r *np.random.random(dim)* levy1(dim) * positions[i, :])
                X1 = gbest - A1 * D_alpha
  
                
                r1 = np.random.random(dim)
                A2 = 2 * a * r1 - a
                D_beta= abs(lbest - r*np.random.random(dim)*levy1(dim) * positions[i, :])
                X2 = lbest - A2 * D_beta

                
                r1 = np.random.random(dim)
                A3 = 2 * a * r1 - a
                D_delta= abs(pbest[i,:] - r*np.random.random(dim)*levy1(dim) *positions[i, :])
                X3 = pbest[i,:] - A3 * D_delta


            minfit = obj_func(X1)
            minp = X1.copy()


            if obj_func(X2) < minfit:
                minp = X2.copy()
                minfit = obj_func(X2)
            elif obj_func(X3) < minfit:
                minp = X3.copy()
                minfit = obj_func(X3)
            minp1 = np.zeros(dim)
            minp1 = minp + np.random.normal(0,abs(F)*np.random.random()*np.cos(t/max_iter * np.pi/2))
            if obj_func(minp1) < minfit:
                minp = minp1
                                                             

            positions[i,:] = minp
            positions[i,:] = np.clip(positions[i], lb, ub)
            fitness[i] = obj_func(positions[i,:])
            if fitness[i] < fit_pbest[i]:
                pbest[i,:] = positions[i,:]
                fit_pbest[i] = fitness[i]
        
        sorted_idx = np.argsort(fitness)
        lbest = positions[sorted_idx[0],:].copy()
        fit_lbest = fitness[sorted_idx[0]]
        if fit_lbest < fit_gbest:
            gbest = lbest
            fit_gbest = fit_lbest
        Curve[t] = fit_gbest

    return fit_gbest, gbest, Curve
