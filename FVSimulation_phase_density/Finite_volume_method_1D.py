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
    self.D0_bTa = 75 #um2 us-1
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

class KID_params():
  def __init__(self,eta_pb,sigma_IC,Teff=False,Q0=False,KID=False,lambda_ph=False,tau_ringing=False,dthetadN=False,N0=False,D0=False,L=False,Delta=False,length=False,height=False,width=False):
    # Copy data from KID. If optional param is given, overwrite KID value with individually specified value.
    if KID == False:
      print('Warning! No KID data given. Make sure to check whether all other optional parameters are provided, otherwise garbage may be produced in the simulation.')
    
    self.eta_pb = eta_pb
    self.sigma_IC = sigma_IC
    self.lambda_ph = lambda_ph if (KID==False) or (lambda_ph!=False) else KID.lambda_ph
    self.tau_ringing = tau_ringing if (KID==False) or (tau_ringing!=False) else KID.tau_ringing
    self.dthetadN = dthetadN if (KID==False) or (dthetadN!=False) else KID.dthetadN
    self.N0 = N0 if (KID==False) or (N0!=False) else KID.N0
    self.D0 = D0 if (KID==False) or (D0!=False) else KID.D0
    self.L = L if (KID==False) or (L!=False) else KID.L
    self.Delta = Delta if (KID==False) or (Delta!=False) else KID.Delta
    self.length = length if (KID==False) or (length!=False) else KID.length
    self.height = height if (KID==False) or (height!=False) else KID.height
    self.width = width if (KID==False) or (width!=False) else KID.width

    if (Teff==False) and (Q0==False):
      raise ValueError('Either Teff or Q0 should be specified.')
    self.Q0 = Q0 if (Q0!=False) else self.T_to_nqp(Teff)

  def T_to_nqp(self,Teff):
    return 2*self.N0*np.sqrt(2*np.pi*consts.k_B*Teff*self.Delta)*np.exp(-self.Delta/(consts.k_B*Teff))*self.height*self.width
  
  def print(self):
    print('eta_pb: \t',self.eta_pb)
    print('sigma_IC: \t',self.sigma_IC)
    print('Q0: \t\t', self.Q0)
    print('lambda_ph: \t',self.lambda_ph)
    print('tau_ringing: \t',self.tau_ringing)
    print('dthetadN: \t',self.dthetadN)
    print('N0: \t\t',self.N0)
    print('D0: \t\t',self.D0)
    print('L: \t\t',self.L)
    print('Delta: \t\t',self.Delta)
    print('lxhxw: \t\t',self.length,'x',self.height,'x',self.width)


class KID_sim():
  def __init__(self,params,dt_init,dx_or_fraction,dt_max=10,simtime_approx=100,method='CrankNicolson',adaptivedx=True,adaptivedt=True,usesymmetry=True,D_const=False,dt_interp=0.01):
    #copy physical parameters from params object
    self.lambda_ph = params.lambda_ph
    self.sigma_IC = params.sigma_IC
    self.eta_pb = params.eta_pb #fit
    self.Q0 = params.Q0 #fit
    self.tau_ringing = params.tau_ringing
    self.dthetadN = params.dthetadN #fit
    self.N0 = params.N0
    self.D0 = params.D0
    self.L = params.L
    self.Delta = params.Delta
    self.length = params.length
    self.height = params.height
    self.width = params.width

    self.K = self.L/(2*self.Q0)
    
    self.E_ph = consts.h*consts.c/self.lambda_ph
    self.Nqp_init = self.eta_pb*self.E_ph/self.Delta

    #simulation settings
    if method == 'BackwardEuler': #more stable
      self.step = self.backwardeuler_step
    elif method == 'CrankNicolson': #more accurate
      self.step = self.CN_step
    else:
      raise ValueError('Invalid option')
    self.adaptivedx = adaptivedx #Increase the courseness of the grid according to the expected width of the distribution. => also sets dx = sigma_IC*dx_or_fraction
    self.usesymmetry = usesymmetry #Simulates only half the domain, efficient for a symmetrical situation
    self.adaptivedt = adaptivedt #Keeps the product of dt and DN constant at each step, such that for smaller expected changes in N (in the tail of the decay), larger timesteps are used.

    dt = dt_init
    self.t_axis=[0]
    self.dtlist=[dt_init]

    # based on settings, decide on geometry.
    if self.adaptivedx: 
      dx = self.sigma_IC*dx_or_fraction
    else:
      dx = dx_or_fraction
    if self.usesymmetry:
      maxdiv = int(np.ceil(self.length/2/dx))
      valid_dx_list = self.length/2/np.arange(1,maxdiv+0.5)[::-1]
    else:
      maxdiv = int(np.ceil(self.length/dx))
      valid_dx_list = self.length/np.arange(1,maxdiv+0.5)[::-1]
    dx = valid_dx_list[0]
    x_borders , x_centers = self.set_geometry(dx,self.length)

    # initialize output arrays
    self.Qintime = [np.exp(-0.5*(x_centers/self.sigma_IC)**2)*self.Nqp_init/(self.sigma_IC*np.sqrt(2*np.pi))] # IC
    self.Qintime[0] = self.Qintime[0]*self.Nqp_init/self.integrate(self.Qintime[0],dx)
    self.Nqpintime = [self.integrate(self.Qintime[0],dx)]
    self.dxlist = [dx]
    self.x_centers_list = [x_centers]

    # calc thermal density of quasiparticles
    Teff_thermal = self.nqp_to_T(self.Q0,self.N0,self.Delta,self.height,self.width)
    Dfinal = self.D0*np.sqrt(2*consts.k_B*Teff_thermal/(np.pi*self.Delta))

    # run simulation
    i=0
    t_elapsed=0

    with tqdm(total=simtime_approx+2, bar_format='{l_bar}{bar}| time (us): {n_fmt}') as pbar:
      while True: # do-while loop
        # handle adaptive dx
        if self.adaptivedx and (i!=0) and (dx!=valid_dx_list[-1]):
          sqrtMSD = np.sqrt(2*Dfinal*t_elapsed)+self.sigma_IC
          dx = valid_dx_list[valid_dx_list <= sqrtMSD*dx_or_fraction][-1]
          x_borders, x_centers = self.set_geometry(dx,self.length)
          Qprev = np.interp(x_centers,x_centersprev,self.Qintime[i])
          Qprev *= self.integrate(self.Qintime[i],self.dxlist[i])/self.integrate(Qprev,dx)
        else:
          Qprev = self.Qintime[i]

        # update diffusion
        if D_const:
          D=self.D0
        else:
          Teff_x = self.nqp_to_T(Qprev+self.Q0,self.N0,self.Delta,self.height,self.width)
          D = self.calc_D(self.D0,Teff_x,self.Delta,x_borders,x_centers)

        # do simulation step
        self.dxlist.append(dx)
        self.dtlist.append(dt)
        self.Qintime.append(self.step(dt,dx,D,self.L,self.K,Qprev))
        self.Nqpintime.append(self.integrate(self.Qintime[i+1],dx))
        self.x_centers_list.append(x_centers)

        pbar.update(dt)
        x_centersprev = x_centers
        t_elapsed+=dt
        self.t_axis.append(t_elapsed)
        if t_elapsed>simtime_approx:
          break

        # handle adaptive dt
        if self.adaptivedt and (dt<=dt_max):
          dN = self.Nqpintime[i]-self.Nqpintime[i+1]
          if i==0:
            dNdt = dN*dt
          if dN != 0:
            dt = dNdt/dN
        i+=1

    self.t_axis = np.array(self.t_axis)
    self.Nqpintime=np.array(self.Nqpintime)

    self.t_axis_interp = np.arange(0,self.t_axis[-1],dt_interp)
    Nqp_interp = np.interp(self.t_axis_interp,self.t_axis,self.Nqpintime)
    self.phaseintime = self.ringing(self.t_axis_interp,Nqp_interp*self.dthetadN,self.tau_ringing)
    self.t_start = -self.t_axis_interp[np.argmax(self.phaseintime)]
    self.t_axis_interp += self.t_start

  def set_geometry(self,dx,length):
    if self.usesymmetry:
      x_borders=np.arange(0,length/2+dx/2,dx)
      x_centers=np.arange(dx/2,length/2,dx)
    else:
      x_borders=np.arange(-length/2,length/2+dx/2,dx)
      x_centers=np.arange(-length/2+dx/2,length/2,dx)
    return x_borders,x_centers

  def nqp_to_T(self,nqp,N0,Delta,height,width):
    a = 2*N0*height*width*np.sqrt(2*np.pi*consts.k_B*Delta)
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
  
  def ringing(self,t_axis,phaseintime,tau_ringing):
    lenT = len(t_axis)
    padded = np.pad(phaseintime,(lenT,lenT),constant_values=(0,0))
    convring = np.exp(-t_axis/tau_ringing)
    convring /= np.sum(convring)
    return np.convolve(padded,convring,'valid')[:lenT]