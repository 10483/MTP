import numpy as np
from scipy.optimize import fsolve
from scipy.signal import deconvolve
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
  def __init__(self,chippath,lambda_ph_in_nm,KIDno,readout_power,temp_in_mK,length,T_eff,N0,sigma_IC,D):
    #copy stuff
    self.chippath = chippath
    self.lambda_ph_in_nm = lambda_ph_in_nm
    self.lambda_ph=lambda_ph_in_nm/1000
    self.KIDno = KIDno
    self.readout_power=readout_power
    self.temp_in_mK = temp_in_mK
    self.temp=temp_in_mK/1000
    self.length=length
    self.T_eff=T_eff
    self.N0=N0
    self.sigma_IC=sigma_IC
    self.D=D
    #get time series
    self.getpulsedata()
    #get resonator data
    self.getresdata()
    #calculate dNqp
    self.ringing = np.exp(-self.t_full/self.tau_ringing)
    self.ringing /= np.sum(self.ringing)
    self.dNqp_phase, _ = deconvolve(np.pad(self.phase/self.dthetadN,(0,len(self.t_full)-1),constant_values=0),self.ringing)
    self.dNqp_amp, _ = deconvolve(np.pad(self.amp/self.dAdN,(0,len(self.t_full)-1),constant_values=0),self.ringing)
    #calculate other stuff
    self.E_ph=consts.h*consts.c/self.lambda_ph
    self.Delta = 1.764*consts.k_B*self.T_c
    self.dNqp_init = np.max(self.dNqp_phase)
    self.eta_pb = self.dNqp_init/(self.E_ph/self.Delta) # or: np.max(self.dNqp_amp)/(self.E_ph/self.Delta)
    self.nqp_thermal = 2*self.N0*np.sqrt(2*np.pi*consts.k_B*self.T_eff*self.Delta)*np.exp(-self.Delta/(consts.k_B*self.T_eff))*self.width*self.height

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
    self.F0=self.Tdepdata[argT,5]*1e-6 #convert to /us from Hz
    self.T_c=self.Tdepdata[argT,21]
    self.volume = self.Tdepdata[argT,14]
    self.height = self.Tdepdata[argT,25]
    self.width = self.volume/self.height/self.length
    self.dthetadN=self.Tdepdata[argT,10]
    self.dAdN=self.Tdepdata[argT,18]
    self.tau_ringing=self.Quality/(np.pi*self.F0)

  def fit_tail(self,start=0,end=-1,showplots=True):
    self.dNqpfit = self.dNqp_phase[start:end]
    self.t_fit = np.arange(len(self.dNqpfit))
    self.fitpars,self.fitcov=np.polyfit(self.t_fit,np.log(self.dNqpfit),1,cov=True)
    a=np.exp(self.fitpars[1])
    b=self.fitpars[0]
    b_std=np.sqrt(np.diag(self.fitcov)[0])
    self.tauqpstar=-1/b
    self.tauqpstarstd=b_std/b**2
    self.Rprime = 1/(2*self.tauqpstar*self.nqp_thermal)
    if showplots:
      plt.figure()
      plt.semilogy(self.t_fit,self.dNqpfit)
      plt.semilogy(self.t_fit,a*np.exp(-self.t_fit/self.tauqpstar))
      plt.show()

class KID_sim():
  def __init__(self,KID,dt,dx_or_fraction,simtime_approx=100,method='CrankNicolson',adaptivedx=True,usesymmetry=True):
    
    #settings
    if method == 'BackwardEuler': #more stable
      self.step = self.backwardeuler_step
    elif method == 'CrankNicolson': #more accurate
      self.step = self.CN_step
    else:
      raise ValueError('Invalid option')
    
    self.adaptivedx = adaptivedx #Increase the courseness of the grid according to the expected width of the distribution. => also sets dx = KID.sigma_IC*dx_or_fraction
    self.usesymmetry = usesymmetry #Simulate only half the domain

    tsteps = int(np.round(simtime_approx/dt))
    self.t_axis = np.arange(0,dt*(tsteps+0.5),dt)

    if self.adaptivedx: 
      dx = KID.sigma_IC*dx_or_fraction
    else:
      dx = dx_or_fraction
    
    if self.usesymmetry:
      maxdiv = int(np.ceil(KID.length/2/dx))
      valid_dx_list = KID.length/2/np.arange(1,maxdiv+0.5)[::-1]
    else:
      maxdiv = int(np.ceil(KID.length/dx))
      valid_dx_list = KID.length/np.arange(1,maxdiv+0.5)[::-1]
    dx = valid_dx_list[0]

    _ , x_centers = self.set_geometry(dx,KID.length)
    self.timeseriesQ = np.zeros((tsteps+1,len(x_centers)))
    self.timeseriesQ[0] = np.exp(-0.5*(x_centers/KID.sigma_IC)**2)*KID.dNqp_init/(KID.sigma_IC*np.sqrt(2*np.pi)) #IC
    lengthlist = np.zeros(tsteps+1,dtype=int)
    lengthlist[0] = len(x_centers)
    dxlist = np.zeros(tsteps+1)
    dxlist[0] = dx

    for i in tqdm(range(tsteps)):
      if self.adaptivedx and (i!=0) and (dx!=valid_dx_list[-1]):
        sqrtMSD = np.sqrt(2*KID.D*dt*i)+KID.sigma_IC
        dx = valid_dx_list[valid_dx_list < sqrtMSD*dx_or_fraction][-1]
        Qprev = np.interp(x_centers,x_centersprev,self.timeseriesQ[i,:lengthlist[i]])
        Qprev *= self.nqp_to_Nqp(self.timeseriesQ[i],dxlist[i])/self.nqp_to_Nqp(Qprev,dx)
      else:
        Qprev = self.timeseriesQ[i,:lengthlist[i]]
      lengthlist[i+1]=len(x_centers)
      dxlist[i+1]=dx
      self.timeseriesQ[i+1,:lengthlist[i+1]] = self.step(dt,dx,KID.D,KID.Rprime,KID.nqp_thermal,Qprev)
      x_centersprev = x_centers
    
    self.timeseriesNqp = self.nqp_to_Nqp(self.timeseriesQ,dxlist)

  def set_geometry(self,dx,length):
    #x_borders=np.arange(-length/2,length/2+dx/2,dx)
    #x_centers=np.arange(-length/2+dx/2,length/2,dx)
    if self.usesymmetry:
      x_borders=np.arange(0,length/2+dx/2,dx)
      x_centers=np.arange(dx/2,length/2,dx)
    else:
      x_borders=np.arange(-length/2,length/2+dx/2,dx)
      x_centers=np.arange(-length/2+dx/2,length/2,dx)
    return x_borders,x_centers
  def diffuse(self,dx,D,Q_prev):
    Q_temp = np.pad(Q_prev,(1,1),'edge') #Assumes von Neumann BCs, for Dirichlet use e.g. np.pad(Q_prev,(1,1),'constant', constant_values=(0, 0))
    gradient = D*np.diff(Q_temp)/dx
    return (-gradient[:-1]+gradient[1:])/dx
  def backwardeuler_eqs(self,dt,dx,D,Rprime,Q0,Q_prev,Q_next):
    return Q_prev - Q_next + dt*(self.diffuse(dx,D,Q_next) - Rprime*Q_next**2 - 2*Rprime*Q0*Q_next)
  def backwardeuler_step(self,dt,dx,D,Rprime,Q0,Q_prev):
    return fsolve(lambda Q_next : self.backwardeuler_eqs(dt,dx,D,Rprime,Q0,Q_prev,Q_next), Q_prev)
  def CN_eqs(self,dt,dx,D,Rprime,Q0,Q_prev,Q_next):
    return Q_prev - Q_next + 0.5*dt*(self.diffuse(dx,D,Q_next) - Rprime*Q_next**2 - 2*Rprime*Q0*Q_next +
                                     self.diffuse(dx,D,Q_prev) - Rprime*Q_prev**2 - 2*Rprime*Q0*Q_prev)
  def CN_step(self,dt,dx,D,Rprime,Q0,Q_prev):
    return fsolve(lambda Q_next : self.CN_eqs(dt,dx,D,Rprime,Q0,Q_prev,Q_next), Q_prev)

  def nqp_to_Nqp(self,Q,dx):
    if self.usesymmetry:
      Nqp = np.sum(Q,axis=-1)*dx*2
    else:
      Nqp = np.sum(Q,axis=-1)*dx
    return Nqp

class sim_data_comp:
  def __init__(self,KID,SIM):
    self.t_full=KID.t_full
    self.t_sim=SIM.t_axis
    self.Nqp_phase_data = KID.dNqp_phase
    self.Nqp_amp_data = KID.dNqp_amp
    self.Nqp_sim = SIM.timeseriesNqp
    self.alignmaxphase()
  
  def alignmaxphase(self):
    self.tmaxdata = self.t_full[np.argmax(self.Nqp_phase_data)]
    self.t_sim_aligned = self.t_sim + self.tmaxdata