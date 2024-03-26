import numpy as np
from scipy.optimize import fsolve
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import re

# everything in um, us, K and eV  and combinations of them unless stated differently

#define useful constants in consts object
class fixed_values:
  def __init__(self,h,c,k_B,eta_pb_max):
    self.h=h
    self.c=c
    self.k_B=k_B
    self.eta_pb_max=eta_pb_max
consts = fixed_values(4.135667e-9,299792458,8.617343e-5,0.59)

#gather data from specific KID necessary for simulation and data comparison.
class KID_data:
  def __init__(self,chippath,lambda_ph_in_nm,KIDno,readout_power,temp_in_mK,width,length,T_eff,N0,sigma_IC,D):
    #copy stuff
    self.chippath = chippath
    self.lambda_ph_in_nm = lambda_ph_in_nm
    self.lambda_ph=lambda_ph_in_nm/1000
    self.KIDno = KIDno
    self.readout_power=readout_power
    self.temp_in_mK = temp_in_mK
    self.temp=temp_in_mK/1000
    self.width=width
    self.length=length
    self.T_eff=T_eff
    self.N0=N0
    self.sigma_IC=sigma_IC
    self.D=D
    #get time series
    self.getpulsedata()
    #get resonator data
    self.getresdata()
    #calculate stuff
    self.calc()

  def getpulsedata(self):
    self.datapath = self.chippath+str(self.lambda_ph_in_nm)+'nm/KID'+str(self.KIDno)+'_'+str(self.readout_power)+'dBm'+'__TmK'+str(self.temp_in_mK)+'_avgpulse_ampphase.csv'
    self.data = np.genfromtxt(self.datapath,skip_header=1,delimiter=',')
    self.amp = self.data[:,0]
    self.ampstd = self.data[:,1]
    self.phase = self.data[:,2]
    self.phasestd = self.data[:,3]
    self.t_full = np.arange(len(self.phase))

  def getresdata(self):
    self.Tdeppath = self.chippath+'S21/2D/KID'+str(self.KIDno)+'_'+str(self.readout_power)+'dBm_Tdep.csv'
    if os.path.exists(self.Tdeppath):
      self.Tdepdata = np.genfromtxt(self.Tdeppath,skip_header=1,delimiter=',')
    else:
      filenames = os.listdir(self.chippath+'S21/2D/')
      pattern = re.compile(r'KID'+str(self.KIDno)+r'.*Tdep.*')
      candidates = [filename for filename in filenames if pattern.match(filename)]
      powers = np.array([int(re.findall(r'\d+',name)[1]) for name in candidates])
      closest = powers[np.argmin(np.abs(powers-self.readout_power))]
      self.Tdeppath = self.chippath+'S21/2D/KID'+str(self.KIDno)+'_'+str(closest)+'dBm_Tdep.csv'
      self.Tdepdata = np.genfromtxt(self.Tdeppath,skip_header=1,delimiter=',')
      print('Readout power not found, instead taking resonator data at closest readout power available.\npath: ', self.Tdeppath)
    TdepT=self.Tdepdata[:,1]
    argT = np.abs(TdepT-self.temp).argmin()
    self.Quality=self.Tdepdata[argT,2]
    self.F0=self.Tdepdata[argT,5]*1e-6 #convert to /us 
    self.T_c=self.Tdepdata[argT,21]
    self.height=self.Tdepdata[argT,25]
    self.dthetadN=self.Tdepdata[argT,10]
    self.dAdN=self.Tdepdata[argT,18]
    self.tau_ringing=self.Quality/(np.pi*self.F0)

  def calc(self):
    self.E_ph=consts.h*consts.c/self.lambda_ph
    self.Delta = 1.764*consts.k_B*self.T_c
    self.eta_pb = (np.max(self.phase)/self.dthetadN)/(self.E_ph/self.Delta)
    self.dNqp_init = self.eta_pb*self.E_ph/self.Delta
    self.nqp_thermal = 2*self.N0*np.sqrt(2*np.pi*consts.k_B*self.T_eff*self.Delta)*np.exp(-self.Delta/(consts.k_B*self.T_eff))*self.width*self.height

  def fit_tail(self,start=0,end=-1,showplots=True):
    self.phasefit = self.phase[start:end]
    self.t_fit = np.arange(len(self.phasefit))
    self.fitpars,self.fitcov=np.polyfit(self.t_fit,np.log(self.phasefit),1,cov=True)
    a=np.exp(self.fitpars[1])
    b=self.fitpars[0]
    b_std=np.sqrt(np.diag(self.fitcov)[0])
    self.tauqpstar=-1/b
    self.tauqpstarstd=b_std/b**2
    self.Rprime = 1/(2*self.tauqpstar*self.nqp_thermal)
    if showplots:
      plt.figure()
      plt.semilogy(self.t_fit,self.phasefit)
      plt.semilogy(self.t_fit,a*np.exp(-self.t_fit/self.tauqpstar))
      plt.show()

class KID_sim():
  def __init__(self,KID,dt,dx,simtime_approx=100,method='CrankNicolson',ring=True,gaussianIC=True,useSymmetry=True):
    if method == 'RK4':
      self.step = self.RK4_step
    elif method == 'ForwardEuler':
      self.step = self.forwardeuler_step
    elif method == 'BackwardEuler':
      self.step = self.backwardeuler_step
    elif method == 'CrankNicolson':
      self.step = self.CN_step
    else:
      raise ValueError('Invalid option')
    
    self.dt=dt
    self.dx=dx
    self.steps = int(np.round(simtime_approx/dt))
    self.t_axis = np.arange(0,dt*(self.steps+0.5),dt)
    self.simtime = self.t_axis[-1]
    self.ring=ring
    self.gaussianIC=gaussianIC
    self.useSymmetry = useSymmetry

    self.simulate(KID)

  def set_geometry(self,length,D):
    if self.useSymmetry:
      self.x_borders=np.arange(0,length/2+self.dx/2,self.dx)
      self.x_centers=np.arange(self.dx/2,length/2+self.dx/2,self.dx)
    else:
      self.x_borders=np.arange(-length/2,length/2+self.dx/2,self.dx)
      self.x_centers=np.arange(-length/2+self.dx/2,length/2+self.dx/2,self.dx)
    self.D=np.ones_like(self.x_borders)*D

  def set_IC(self,sigma_IC,dNqp_init):
    if self.gaussianIC:
      self.IC=np.exp(-0.5*(self.x_centers/sigma_IC)**2)*dNqp_init/(sigma_IC*np.sqrt(2*np.pi)) 
    else:
      self.IC=np.zeros_like(self.x_centers)
      if self.useSymmetry:
        self.IC[0]=dNqp_init/2/self.dx
      else:
        self.IC[int(round(len(self.x_centers)/2))]=dNqp_init/self.dx

  def diffuse(self,Q_prev):
    Q_temp = np.pad(Q_prev,(1,1),'edge') #Assumes von Neumann BCs, for Dirichlet use e.g. np.pad(Q_prev,(1,1),'constant', constant_values=(0, 0))
    gradient = self.D*np.diff(Q_temp)/self.dx
    return (-gradient[:-1]+gradient[1:])/self.dx

  def forwardeuler_step(self,Rprime,Q0,Q_prev):
    Q_next=Q_prev.copy()
    # simple terms
    Q_next -= self.dt*Rprime*(Q_prev**2+2*Q0*Q_prev)
    # diffusion
    Q_next += self.dt*self.diffuse(Q_prev)
    return Q_next

  def rhs_estimate(self,Rprime,Q0,Q_prev):
    rhs=np.zeros_like(Q_prev)
    # simple terms
    rhs -= Rprime*(Q_prev**2+2*Q0*Q_prev)
    # diffusion
    rhs += self.diffuse(Q_prev)
    return rhs

  def RK4_step(self,Rprime,Q0,Q_prev):
    k1 = self.rhs_estimate(Rprime,Q0,Q_prev)
    k2 = self.rhs_estimate(Rprime,Q0,Q_prev+k1*self.dt/2)
    k3 = self.rhs_estimate(Rprime,Q0,Q_prev+k2*self.dt/2)
    k4 = self.rhs_estimate(Rprime,Q0,Q_prev+k3*self.dt)
    Q_next = Q_prev + (k1+2*k2+2*k3+k4)*self.dt/6
    return Q_next

  def backwardeuler_eqs(self,Rprime,Q0,Q_prev,Q_next):
    return Q_prev - Q_next + self.dt*(self.diffuse(Q_next) - Rprime*Q_next**2 - 2*Rprime*Q0*Q_next)

  def backwardeuler_step(self,Rprime,Q0,Q_prev):
    return fsolve(lambda Q_next : self.backwardeuler_eqs(Rprime,Q0,Q_prev,Q_next), Q_prev)

  def CN_eqs(self,Rprime,Q0,Q_prev,Q_next):
    return Q_prev - Q_next + 0.5*self.dt*(self.diffuse(Q_next) - Rprime*Q_next**2 - 2*Rprime*Q0*Q_next +
                                          self.diffuse(Q_prev) - Rprime*Q_prev**2 - 2*Rprime*Q0*Q_prev)

  def CN_step(self,Rprime,Q0,Q_prev):
    return fsolve(lambda Q_next : self.CN_eqs(Rprime,Q0,Q_prev,Q_next), Q_prev)

  def simulate(self,KID):
    self.set_geometry(KID.length,KID.D)
    self.set_IC(KID.sigma_IC,KID.dNqp_init)
    self.timeseriesQ = np.zeros((self.steps+1,len(self.IC)))
    self.timeseriesQ[0,:]=self.IC
    for i in tqdm(range(self.steps)):
      self.timeseriesQ[i+1,:]=self.step(KID.Rprime,KID.nqp_thermal,self.timeseriesQ[i,:])
    
    self.nqp_to_Nqp()
    if self.ring==True:
      self.signal = self.ringing(KID.tau_ringing)
    else:
      self.signal = self.timeseriesNqp
    self.phase=KID.dthetadN*self.signal
    self.amp=KID.dAdN*self.signal

  def nqp_to_Nqp(self):
    if self.useSymmetry:
      self.timeseriesNqp = np.trapz(self.timeseriesQ,self.x_centers,axis=1)*2
    else:
      self.timeseriesNqp = np.trapz(self.timeseriesQ,self.x_centers,axis=1)

  def ringing(self,tau_ringing):
    lenT = len(self.t_axis)
    padded = np.pad(self.timeseriesNqp,(lenT,lenT),constant_values=(0,0))
    convring = np.exp(-self.t_axis/tau_ringing)
    convring /= np.sum(convring)
    return np.convolve(padded,convring,'valid')[:lenT]

class sim_data_comp:
  def __init__(self,KID,SIM):
    self.t_full=KID.t_full
    self.t_sim=SIM.t_axis
    self.dt=SIM.dt
    self.phasedata = KID.phase
    self.ampdata = KID.amp
    self.phasesim = SIM.phase
    self.ampsim = SIM.amp
    self.alignmaxphase()
  
  def alignmaxphase(self):
    self.tmaxdata = self.t_full[np.argmax(self.phasedata)]
    self.tmaxsim = self.t_sim[np.argmax(self.phasesim)]
    self.t_sim_aligned = self.t_sim + (self.tmaxdata-self.tmaxsim)