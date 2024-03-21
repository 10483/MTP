import numpy as np
from scipy.optimize import fsolve
from tqdm import tqdm
import matplotlib.pyplot as plt

# everything in um, us, K and eV  and combinations of them unless stated differently

#define useful constants object
class fixed_values:
  def __init__(self,h,c,k_B,eta_pb_max)
    self.h=h
    self.c=c
    self.k_B=k_B
    self.eta_pb_max=eta_pb_max
consts = fixed_values(4.1356e-9,299792458,0.00008617343,0.59)

#gather data from specific KID necessary for simulation and data comparison.
class KID_data:
  def __init__(self,chippath,lambda_ph_in_nm,KIDno,readout_power,temp,width,length,T_eff,N0,sigma_IC,D,eta_pb=consts.eta_pb_max):
    #copy stuff
    self.lambda_ph=lambda_ph_in_nm/1000
    self.KIDno = KIDno
    self.readout_power=readout_power
    self.temp=temp
    self.width=width
    self.length=length
    self.T_eff=T_eff
    self.N0=N0
    self.sigma_IC=sigma_IC
    self.D=D
    self.eta_pb=eta_pb
    #get time series
    self.datapath = chippath+str(lambda_ph_in_nm)+'nm/KID'+str(KIDno)+'_'+str(readout_power)+'dBm'+'__TmK'+str(temp)+'_avgpulse_ampphase.csv'
    self.data = np.genfromtxt(self.datapath,skip_header=1,delimiter=',')
    self.amp = self.data[:,0]
    self.ampstd = self.data[:,1]
    self.phase = self.data[:,2]
    self.phasestd = self.data[:,3]
    self.t_full = np.arange(len(self.phase))
    #get resonator data
    self.Tdeppath = chippath+'S21/2D/KID'+str(KIDno)+'_'+str(readout_power)+'dBm_Tdep.csv'
    Tdepdata = np.genfromtxt(self.Tdeppath,skip_header=1,delimiter=',')
    TdepT=Tdepdata[:,1]
    argT = np.abs(TdepT-self.temp).argmin()
    self.Quality=Tdepdata[argT,2]
    self.F0=Tdepdata[argT,5]
    self.T_c=Tdepdata[argT,21]
    self.height=Tdepdata[argT,25]
    self.dthetadN=Tdepdata[argT,10]
    self.dAdN=Tdepdata[argT,18]
    self.tau_ringing=self.Quality/(np.pi*self.F0)
    #calculate stuff
    self.E_ph=consts.h*consts.c/self.lambda_ph
    self.Delta = 1.764*consts.k_B*self.T_c
    self.dNqp_init = self.eta_pb*self.E_ph/self.Delta
    self.Nqp_thermal = 2*N0*np.sqrt(2*np.pi*consts.k_B*T_eff*self.Delta)*np.exp(-self.Delta/(consts.k_B*T_eff))*width*self.height
  
  def fit_tail(self,start=0,end=-1):
    self.phasefit = self.phase[start:end]
    self.t_fit = np.arange(len(self.phasefit))
    self.fitpars,self.fitcov=np.polyfit(self.t_fit,np.log(self.phasefit),1,cov=True)
    a=np.exp(self.fitpars[1])
    b=self.fitpars[0]
    b_std=np.sqrt(np.diag(self.fitcov)[0])
    self.tauqpstar=-1/b
    self.tauqpstarstd=b_std/b**2
    self.Rprime = 1/(2*self.tauqpstar*self.Nqp_thermal)
    plt.figure()
    plt.semilogy(self.t_fit,self.phasefit)
    plt.semilogy(self.t_fit,a*np.exp(-self.t_fit/self.tauqpstar))

class simulator():
  def __init__(self,dt,dx,simtime_approx=100,method='CrankNicolson'):
    self.method=method
    self.dt=dt
    self.dx=dx
    self.steps = int(np.round(simtime_approx/dt))
    self.t_axis = np.arange(0,dt*(self.steps+1),dt)
    self.simtime = self.t_axis[-1]

  def set_geometry(self):



  def set_IC(self):
    self.IC=
  
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

  def simulate(Q_start,steps,dt,dx,D,R,Q0,method='CrankNicolson'):
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
    

    return t_list, Q_list

  def nqp_to_Nqp(Q_list,x_centers):
    return np.trapz(Q_list,x_centers,axis=1)

  def Nqp_to_theta(Nqp,dthetadN,t_axis):
    theta = Nqp*dthetadN
    lenT = len(theta)
    thetapadded = np.pad(theta,(lenT,lenT),constant_values=(0,0))
    Ringing = np.exp(-t_axis/tau_r)
    Ringing/=np.sum(Ringing)
    return np.convolve(thetapadded,Ringing,'valid')[:lenT]
