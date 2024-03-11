import numpy as np
from scipy.optimize import fsolve
from tqdm import tqdm

def diffuse(dx,D,Q_prev):
  Q_temp = np.pad(Q_prev,(1,1),'edge') #Assumes von Neumann BCs, for Dirichlet use e.g. np.pad(Q_prev,(1,1),'constant', constant_values=(0, 0))
  gradient = D*np.diff(Q_temp)/dx
  return (-gradient[:-1]+gradient[1:])/dx

def forwardeuler_step(dt,dx,D,R,Q0,Q_prev):
  Q_next=Q_prev.copy()
  # simple terms
  Q_next -= dt*R*(Q_prev**2+2*Q0*Q_prev)
  # diffusion
  Q_next += dt*diffuse(dx,D,Q_prev)
  return Q_next

def rhs_estimate(dx,D,R,Q0,Q_prev):
  rhs=np.zeros_like(Q_prev)
  # simple terms
  rhs -= R*(Q_prev**2+2*Q0*Q_prev)
  # diffusion
  rhs += diffuse(dx,D,Q_prev)
  return rhs

def RK4_step(dt,dx,D,R,Q0,Q_prev):
  k1 = rhs_estimate(dx,D,R,Q0,Q_prev)
  k2 = rhs_estimate(dx,D,R,Q0,Q_prev+k1*dt/2)
  k3 = rhs_estimate(dx,D,R,Q0,Q_prev+k2*dt/2)
  k4 = rhs_estimate(dx,D,R,Q0,Q_prev+k3*dt)
  Q_next = Q_prev + (k1+2*k2+2*k3+k4)*dt/6
  return Q_next

def backwardeuler_eqs(dt,dx,D,R,Q0,Q_prev,Q_next):
  return Q_prev - Q_next + dt*(diffuse(dx,D,Q_next) - R*Q_next**2 - 2*R*Q0*Q_next)

def backwardeuler_step(dt,dx,D,R,Q0,Q_prev):
  return fsolve(lambda Q_next : backwardeuler_eqs(dt,dx,D,R,Q0,Q_prev,Q_next), Q_prev)

def CN_eqs(dt,dx,D,R,Q0,Q_prev,Q_next):
  return Q_prev - Q_next + 0.5*dt*(diffuse(dx,D,Q_next) - R*Q_next**2 - 2*R*Q0*Q_next +
                                   diffuse(dx,D,Q_prev) - R*Q_prev**2 - 2*R*Q0*Q_prev)

def CN_step(dt,dx,D,R,Q0,Q_prev):
  return fsolve(lambda Q_next : CN_eqs(dt,dx,D,R,Q0,Q_prev,Q_next), Q_prev)

def simulate(Q_start,steps,dt,dx,D,R,Q0,method='BackwardEuler'):
  if method == 'RK4':
    step = RK4_step
  elif method == 'ForwardEuler':
    step = forwardeuler_step
  elif method == 'BackwardEuler':
    step = backwardeuler_step
  elif method == 'CrankNicolson':
    step = CN_step
  else:
    raise ValueError('Invalid option')

  Q_list=np.zeros((steps+1,len(Q_start)))
  Q_list[0,:]=Q_start
  for i in tqdm(range(steps)):
    Q_list[i+1,:]=step(dt,dx,D,R,Q0,Q_list[i,:])
  
  t_list = np.arange(0,dt*(steps+1),dt)
  return t_list, Q_list

def