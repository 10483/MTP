import numpy as np
from scipy.optimize import fsolve
from scipy.special import lambertw
from scipy.signal import deconvolve
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import re

# everything in um, us, K and eV  and combinations of them unless stated differently
class fixed_values:
  def __init__(self):
    self.h=4.135667e-9 #eV us
    self.hbar=self.h/(2*np.pi) #eV us rad-1
    self.c=299792458 #um/us
    self.k_B=8.617343e-5 #eV/K
    self.eta_pb_max=0.59
    self.energy_au=27.211386 #eV
    self.time_au=2.418884e-11 #us
    self.length_au=5.291772e-5 #um
    self.N0_bTa = 30.3e9 #eV-1 um-3
    self.D0_bTa = 50 #um2 us-1
    self.N0_Al = 17.2e9 #eV-1 um-3
    self.D0_Al = 15000 #um2 us-1
consts = fixed_values()

#gather data from specific KID necessary for simulation and data comparison.
class KID_data:
  def __init__(self,chippath,lambda_ph_in_nm,filename,length,samplefreq_in_MHz=1,FFT_power_path=False,material='bTa'):
    #copy stuff
    self.filename = filename
    self.samplefreq_in_MHz = samplefreq_in_MHz
    self.dt = 1/self.samplefreq_in_MHz
    self.chippath = chippath
    self.FFT_power_path = FFT_power_path
    self.lambda_ph_in_nm = lambda_ph_in_nm
    self.lambda_ph = lambda_ph_in_nm/1000
    nums = list(map(int, re.findall(r'\d+', filename)))
    self.KIDno = nums[0]
    self.readout_power=nums[1]
    self.temp_in_mK = nums[2]
    self.temp = self.temp_in_mK/1000
    self.length = length
    if material == 'bTa':
      self.N0 = consts.N0_bTa
      self.D0 = consts.D0_bTa
    elif material == 'Al':
      self.N0 = consts.N0_Al
      self.D0 = consts.D0_Al
    else:
      raise ValueError()
    #get pulse data
    self.getpulsedata()
    #get resonator data
    if self.FFT_power_path==False:
      self.getresdata_S21()
    else:
      self.getresdata_FFT()
    
    '''
    #deconvolve
    self.ringing = np.exp(-self.t_full/self.tau_ringing)
    self.ringing /= np.sum(self.ringing)
    self.phase, _ = deconvolve(np.pad(self.phase,(0,len(self.t_full)-1),constant_values=0),self.ringing)
    self.amp, _ = deconvolve(np.pad(self.amp,(0,len(self.t_full)-1),constant_values=0),self.ringing)
    '''

  def getpulsedata(self):
    self.datapath = self.chippath+str(self.lambda_ph_in_nm)+'nm/'+self.filename
    self.data = np.genfromtxt(self.datapath,skip_header=1,delimiter=',')
    self.amp = self.data[:,0]
    self.ampstd = self.data[:,1]
    self.phase = self.data[:,2]
    self.phasestd = self.data[:,3]
    self.t_full = np.arange(len(self.phase))*self.dt

  def getresdata_S21(self):
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
    self.Delta = 1.768*consts.k_B*self.T_c
  
  def getresdata_FFT(self):
    path_to_data = self.chippath+self.FFT_power_path+'KID'+str(self.KIDno)+'_0dBm__all_td_averaged.dat'
    with open(path_to_data,'r') as file:
        lines = file.readlines()
    line3=lines[2].strip()
    line4=lines[3].strip()
    self.F0 = float(re.findall(r"\d*\.?\d+",line3)[0]) * 1000 #from Ghz to us-1
    self.Quality = float(re.findall(r"\d*\.?\d+",line4)[0])
    self.tau_ringing = self.Quality/(np.pi*self.F0)

  def fit_tail(self,start=0,end=-1,showplots=True):
    self.phasefit = self.phase[start:end]
    self.t_fit = self.t_full[start:end]
    phasefit = self.phasefit[self.phasefit>0]
    t_fit = self.t_fit[self.phasefit>0]
    self.fitpars,self.fitcov=np.polyfit(t_fit,np.log(phasefit),1,cov=True)
    a=np.exp(self.fitpars[1])
    b=self.fitpars[0]
    b_std=np.sqrt(np.diag(self.fitcov)[0])
    self.tauqpstar=-1/b
    self.L=1/self.tauqpstar
    self.tauqpstarstd=b_std/b**2
    if showplots:
      plt.figure()
      plt.semilogy(self.phasefit)
      plt.semilogy(a*np.exp(-self.t_fit/self.tauqpstar))
      plt.show()

class KID_sim():
  def __init__(self,KID,Teff_thermal,K,phi_init,dt,dx_or_fraction,sigma_IC=0.5,simtime_approx=100,method='CrankNicolson',adaptivedx=True,adaptivedt=True,usesymmetry=True,D_const=False):
    # copy variables and process settings
    self.phi_init = phi_init
    self.K = K
    self.Teff_thermal = Teff_thermal

    self.tau_ringing = KID.tau_ringing
    self.dthetadN = KID.dthetadN
    self.D0 = KID.D0
    self.L = KID.L
    
    #settings
    if method == 'BackwardEuler': #more stable
      self.step = self.backwardeuler_step
    elif method == 'CrankNicolson': #more accurate
      self.step = self.CN_step
    else:
      raise ValueError('Invalid option')
    
    self.adaptivedx = adaptivedx #Increase the courseness of the grid according to the expected width of the distribution. => also sets dx = sigma_IC*dx_or_fraction
    self.usesymmetry = usesymmetry #Simulate only half the domain
    self.adaptivedt = adaptivedt # from initial phase to phase/10, ramp up dt from start value to 1 (Not yet implemented)

    tsteps = int(np.round(simtime_approx/dt))
    self.t_axis = np.arange(0,dt*(tsteps+0.5),dt)

    # based on settings, decide on geometry.
    if self.adaptivedx: 
      dx = sigma_IC*dx_or_fraction
    else:
      dx = dx_or_fraction
    if self.usesymmetry:
      maxdiv = int(np.ceil(KID.length/2/dx))
      valid_dx_list = KID.length/2/np.arange(1,maxdiv+0.5)[::-1]
    else:
      maxdiv = int(np.ceil(KID.length/dx))
      valid_dx_list = KID.length/np.arange(1,maxdiv+0.5)[::-1]
    dx = valid_dx_list[0]
    x_borders , x_centers = self.set_geometry(dx,KID.length)

    # initialize output arrays
    self.timeseriesphi = np.zeros((tsteps+1,len(x_centers)))
    self.timeseriesphi[0] = np.exp(-0.5*(x_centers/sigma_IC)**2)*self.phi_init/(sigma_IC*np.sqrt(2*np.pi)) #IC
    self.timeseriesphi[0]=self.timeseriesphi[0]*self.phi_init/self.integrate(self.timeseriesphi[0],dx)
    # initialize arrays necessary for integrating the phase density
    lengthlist = np.zeros(tsteps+1,dtype=int)
    lengthlist[0] = len(x_centers)
    dxlist = np.zeros(tsteps+1)
    dxlist[0] = dx

    # calc thermal density of quasiparticles
    self.Q0 = self.T_to_nqp(Teff_thermal,KID.N0,KID.Delta,KID.height)
    Dfinal = self.D0*np.sqrt(2*consts.k_B*Teff_thermal/(np.pi*KID.Delta))

    # run simulation
    for i in tqdm(range(tsteps)):

      # handle adaptive dx
      if self.adaptivedx and (i!=0) and (dx!=valid_dx_list[-1]):
        sqrtMSD = np.sqrt(2*Dfinal*dt*i)+sigma_IC
        dx = valid_dx_list[valid_dx_list <= sqrtMSD*dx_or_fraction][-1]
        x_borders, x_centers = self.set_geometry(dx,KID.length)
        Qprev = np.interp(x_centers,x_centersprev,self.timeseriesphi[i,:lengthlist[i]])
        Qprev *= self.integrate(self.timeseriesphi[i],dxlist[i])/self.integrate(Qprev,dx)
      else:
        Qprev = self.timeseriesphi[i,:lengthlist[i]]

      # update diffusion
      if D_const:
        D=Dfinal
      else:
        dnqp = Qprev/self.dthetadN
        Teff_x = self.nqp_to_T(dnqp+self.Q0,KID.N0,KID.Delta,KID.height)
        D = self.calc_D(self.D0,Teff_x,KID.Delta,x_borders,x_centers)

      # do simulation step
      lengthlist[i+1]=len(x_centers)
      dxlist[i+1]=dx
      self.timeseriesphi[i+1,:lengthlist[i+1]] = self.step(dt,dx,D,self.L,K,Qprev)
      x_centersprev = x_centers
    
    self.dxlist = dxlist
    self.timeseriestheta_raw = self.integrate(self.timeseriesphi,dxlist)
    self.timeseriestheta = self.ringing(self.tau_ringing)
    self.t_axis -= self.t_axis[np.argmax(self.timeseriestheta)]

  def set_geometry(self,dx,length):
    if self.usesymmetry:
      x_borders=np.arange(0,length/2+dx/2,dx)
      x_centers=np.arange(dx/2,length/2,dx)
    else:
      x_borders=np.arange(-length/2,length/2+dx/2,dx)
      x_centers=np.arange(-length/2+dx/2,length/2,dx)
    return x_borders,x_centers
  
  def T_to_nqp(self,Teff,N0,Delta,height):
    return 2*N0*np.sqrt(2*np.pi*consts.k_B*Teff*Delta)*np.exp(-Delta/(consts.k_B*Teff))*height

  def nqp_to_T(self,nqp,N0,Delta,height):
    a = 2*N0*height*np.sqrt(2*np.pi*consts.k_B*Delta)
    b = Delta/consts.k_B
    return np.real(2*b/lambertw(2*a**2*b/(nqp**2)))

  
  def calc_D(self,D0,Teff_x,Delta,x_borders,x_centers):
    return np.interp(x_borders,x_centers,D0*np.sqrt(2*consts.k_B*Teff_x/(np.pi*Delta)))
  
  def diffuse(self,dx,D,Q_prev):
    Q_temp = np.pad(Q_prev,(1,1),'edge') #Assumes von Neumann BCs, for Dirichlet use e.g. np.pad(Q_prev,(1,1),'constant', constant_values=(0, 0))
    gradient = D*np.diff(Q_temp)/dx
    return (-gradient[:-1]+gradient[1:])/dx
  def backwardeuler_eqs(self,dt,dx,D,L,K,Q_prev,Q_next):
    return Q_prev - Q_next + dt*(self.diffuse(dx,D,Q_next) - K*Q_next**2 - L*Q_next)
  def backwardeuler_step(self,dt,dx,D,L,K,Q_prev):
    return fsolve(lambda Q_next : self.backwardeuler_eqs(dt,dx,D,L,K,Q_prev,Q_next), Q_prev)
  def CN_eqs(self,dt,dx,D,L,K,Q_prev,Q_next):
    return Q_prev - Q_next + 0.5*dt*(self.diffuse(dx,D,Q_next) - K*Q_next**2 - L*Q_next +
                                     self.diffuse(dx,D,Q_prev) - K*Q_prev**2 - L*Q_prev)
  def CN_step(self,dt,dx,D,L,K,Q_prev):
    return fsolve(lambda Q_next : self.CN_eqs(dt,dx,D,L,K,Q_prev,Q_next), Q_prev)

  def integrate(self,Q,dx):
    if self.usesymmetry:
      Nqp = np.sum(Q,axis=-1)*dx*2
    else:
      Nqp = np.sum(Q,axis=-1)*dx
    return Nqp
  
  def ringing(self,tau_ringing):
    lenT = len(self.t_axis)
    padded = np.pad(self.timeseriestheta_raw,(lenT,lenT),constant_values=(0,0))
    convring = np.exp(-self.t_axis/tau_ringing)
    convring /= np.sum(convring)
    return np.convolve(padded,convring,'valid')[:lenT]