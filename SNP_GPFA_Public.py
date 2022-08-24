from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import matplotlib.pyplot as plt
import matplotlib
import GP_fourier as gpf 
import random as rand 


import numpy as npy
import scipy as sp
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm
import autograd.scipy.stats.poisson as poisson
from autograd.extend import primitive, defvjp
from autograd import grad


from autograd import value_and_grad
from autograd.misc.optimizers import adam, sgd
from scipy.optimize import minimize





def softplus(x):
  lt_34 = (x >= 34)
  gt_n37 = (x <= -30.8)
  neither_nor = np.logical_not(np.logical_or(lt_34, gt_n37))
  rval = np.where(gt_n37, 0., x)
  return np.where(neither_nor, np.log(1 + np.exp(x)), rval)


@primitive
def safe_logsoftplus(x, up_limit=30, low_limit = -30):
  x = np.array(x)
  x[x>up_limit] = np.log(x[x>up_limit])
  x[(x<up_limit) & (x>low_limit)] = np.log(softplus(x[(x<up_limit) & (x>low_limit)]))
  #x[x<low_limit] = x[x<low_limit]
  return x
def safe_logsoftplus_vjp(ans, x, low_limit = -30):
  x_shape = x.shape
  operator = np.ones(x.shape)
  operator[x>low_limit] =  1/ ((1+np.exp(-x[x>low_limit]))*softplus(x[x>low_limit]))
  return lambda g: np.full(x_shape,g)* operator
  #return lambda g: np.full(x_shape, g) * 1/ ((1+np.exp(-x))*safe_softplus(x))
defvjp(safe_logsoftplus, safe_logsoftplus_vjp)


def black_box_variational_inference(logprob, N, N_trs, num_samples, dim_sig,dim_noise,n_neurs, nxcirc, wwnrm):
  """Implements http://arxiv.org/abs/1401.0118, and uses the
  local reparameterization trick from http://arxiv.org/abs/1506.02557

  Functionality for this built on autograd demos https://github.com/HIPS/autograd/tree/master/examples"""

  latlength = int(N/(dim_noise*N_trs+dim_sig))
  def unpack_params(params):
      # Variational dist is a diagonal Gaussian.
      n_loadings = (dim_sig+1)*n_neurs
      n_lats_tot = dim_sig + dim_noise

      #unpack all params optimizing over
      mean, log_std= params[:N], params[N:-(n_loadings)-((dim_noise)*n_neurs)-n_lats_tot] #variational params

      #hyper params
      loadings= np.reshape(params[-(n_loadings)-((dim_noise)*n_neurs)-n_lats_tot:-((dim_noise)*n_neurs)-n_lats_tot], [(dim_sig+1), n_neurs])
      C= np.reshape(params[-((dim_noise)*n_neurs)-n_lats_tot:-n_lats_tot], [(dim_noise), n_neurs])
      len_sc_sig = params[-n_lats_tot:-dim_noise] 
      len_sc_noise  = params[-dim_noise:]

      #mean, log_std,rh, len_sc  = params[:N], params[N:-2], params[-2], params[-1]
      return mean, log_std, loadings, C, len_sc_sig, len_sc_noise

  def gaussian_entropy(log_std):
      return 0.5 * N * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

  rs = npr.RandomState(0)
  def variational_objective(params,t):
      """Provides a stochastic estimate of the variational lower bound."""
      mean, log_std, loadings, C, len_sc_sig, len_sc_noise  = unpack_params(params) 
      cdiag_sig = gpf.mkcovs.mkcovdiag_ASD_wellcond(len_sc_sig+0, np.ones(np.size(len_sc_sig)), nxcirc, wwnrm = wwnrm, addition = 1e-7).T #generate cdiag based on len_sc param
      all_cdiag_sig = np.reshape(cdiag_sig.T,[latlength*dim_sig,-1])


      cdiag_noise = gpf.mkcovs.mkcovdiag_ASD_wellcond(len_sc_noise+0, np.ones(np.size(len_sc_noise)), nxcirc, wwnrm = wwnrm, addition = 1e-7).T
      all_cdiag_noise = np.reshape(cdiag_noise.T,[latlength*dim_noise,-1]) #same cdiag for all trials for noise....i don't vary the length scale per trial.

      samples = rs.randn(num_samples, N) * np.exp(log_std) + mean #Generate samples using reparam trick


      lower_bound = gaussian_entropy(log_std) + np.mean(logprob(samples,t, loadings,C, all_cdiag_sig, all_cdiag_noise)) #return elbo and hyperparams (loadings and length scale)

      return -lower_bound

  gradient = grad(variational_objective)

  return variational_objective, gradient, unpack_params


def conv_time_inf(x_samples,lat_length, dim_sig, dim_noise, W, C, Bf, N_trials):
  '''
  Takes in the BBVI sampled latent (or MAP latent) and restructures things, converts to time domain, returns rates to be used
  for the log-joint calculations 

  returned size is samples x latents x N (time domain N)

  Current implementation incorporates the DC offset here. Should be stored in matrices C and W only!


  Notes for potential speed up -- convert to Jax and use jax specific batching instead of hand-coding my own batching
  '''



  dim_sig, n_neurons = W.shape[0]-1, W.shape[1]
  dim_noise = C.shape[0]
  #noise loadings


  numsamps= x_samples.shape[0]



  #signal latents
  x_sig = x_samples[:,:lat_length*dim_sig]
  x_samps_sig = np.reshape(x_sig,[numsamps,dim_sig, lat_length]) 
  time_x_sig = np.matmul(x_samps_sig, Bf[None,:])#convert to time domain (dimension samples x dim sig x N (time domain))

  rates_sig = W[:dim_sig].T@time_x_sig + W[dim_sig][None,:,None]#samples by n_neurons by N


  #pull out noise 
  x_noise = x_samples[:,lat_length*dim_sig:]
  x_noise = np.reshape(x_noise, [numsamps, N_trials, dim_noise, lat_length]) #vectorize along the trials by noise dimension (for matrix operation on C)
  time_x_noise = np.matmul(x_noise,Bf[None,:])#convert to time domain (dimension samples x N_trials x dim noise x N (time domain))
  x_noise_C = np.array([[C[:dim_noise].T@time_x_noise[j,i,:,:]  for i in np.arange(N_trials)] for j in np.arange(numsamps)])#this is samples by n_trials by dim_sig   by N




  return rates_sig, x_noise_C, x_samps_sig,x_noise, numsamps



def log_prob_poiss(x_samples, t,loadings, C, y_train, N_trs, lat_length, dim_sig, dim_noise, Bf, cdiag_sig, cdiag_noise):
  '''
  calculate the log-probability of the model. 
  '''


  rates_sig, x_noise_C, x_samps_sig,x_samps_noise, numsamps = conv_time_inf(x_samples,lat_length, dim_sig, dim_noise, loadings,C,Bf,  N_trs) #transform into time domain


  #rates for all of the neurons
  rates = (x_noise_C) + rates_sig[:,None,:,:] #add activity from the noise and signal subspaces


  #specify axis to sum over
  axes_final = (1,2)
  axes_first = (1,2,3)



  #poisson log-likelihood using softplus nonlinearity
  loglike = -np.sum(np.log(sp.special.factorial(y_train)))+ np.sum(y_train[None,:]*safe_logsoftplus(rates),axis=axes_first) - np.sum(softplus(rates), axis = axes_first)

  #calculate prior
  total_prior = calc_log_prior(x_samps_sig, x_samps_noise,  N_trs, dim_sig, dim_noise, N_four, cdiag_sig,cdiag_noise,numsamps)

  logposts = loglike + total_prior

  return logposts




def calc_log_prior(x_samps_sig,x_samps_noise, N_trs, dim_sig, dim_noise, N_four, cdiag_sig, cdiag_noise, numsamps):
  '''
  Calculates the log prior in the fourier domain.

  need to put a prior over all signal latents and noise latents. Noise latents are number of trials times dimensionality.
  '''

  total_prior = 0

  for i in np.arange(dim_sig):
    x_samp = np.reshape(x_samps_sig[:,i,:],[numsamps,N_four])
    total_prior = total_prior -(1/2)*(np.sum(np.square(x_samp).T/cdiag_sig[i*N_four:(i+1)*N_four],axis=0)+ np.sum(np.log(2*np.pi*cdiag_sig[i*N_four:(i+1)*N_four])))  

  for j in np.arange(N_trs):
    for k in np.arange(dim_noise):
      x_samp = np.reshape(x_samps_noise[:,j,k,:],[numsamps,N_four])
      total_prior = total_prior -(1/2)*(np.sum(np.square(x_samp).T/cdiag_noise[k*N_four:(k+1)*N_four],axis=0)+ np.sum(np.log(2*np.pi*cdiag_noise[k*N_four:(k+1)*N_four])))#only k dims for noise, this will repeat the same cdiag section j times (once per trial)  


  return total_prior



def optimize(y, num_var_params, n_neurs, dim_sig,  dim_noise, N_four, N_trs):

  elbos = []
  hp = []
  times = []

  def callback(params, t, g):
    if t%50 is 0:
      print("Iteration {} lower bound {}".format(t, -varobjective(params, t)))#)
      # len_hats.append(params[-1])
      # loadings_hats.append(params[-(n_latents*n_neurons+1): -1])
      elbos.append(-varobjective(params, t))
      hp.append(params[-(dim_noise+dim_sig):])
      times.append(time.time() - tee)
      print("elapsed time is ....", time.time() - tee)
    return

  #initialize optimization
  init_mean    =  np.zeros(num_var_params)#x_train@Bffts[0].T#0 * npr.randn(N_four)
  init_log_std = -10* np.ones(len(init_mean))
  init_loadings = np.ndarray.flatten(np.zeros((dim_sig+1)*n_neurs))
  init_C = np.ndarray.flatten(np.zeros((dim_noise)*n_neurs))
  init_len_sc = 15*np.ones(dim_sig+dim_noise)
  init_params = np.concatenate([init_mean, init_log_std,init_loadings, init_C, init_len_sc])

  #set optimization params -- annealing BBVI here.
  num_samples = [15,15,15] #number of samples for BBVI (usually don't need too many more than 10 or 15)
  num_iters = [300,300,100] #number of iterations (increase if convergence is not good)
  step_size = [.05,.01,.001] #step size



  logprob= lambda samples, t,loadings,C, cdiag_sig, cdiag_noise : log_prob_poiss(samples, t, loadings,C,y, N_trs, N_four, dim_sig, dim_noise,  Bffts[0],cdiag_sig, cdiag_noise)
  variational_params = init_params

  for i in np.arange(3):
    varobjective, gradient, unpack_params = black_box_variational_inference(logprob, num_var_params, N_trs, num_samples[i], dim_sig,dim_noise,n_neurs, nxcirc, wwnrm) #declare function handles for the derivatives and variational objective
    #natural_gradient = lambda lam, i: (1./fisher_diag(lam, num_var_params)) * gradient(lam, i)
    variational_params =  adam(gradient, variational_params, step_size=step_size[i], num_iters=num_iters[i], callback=callback) #use this as your optimization
    print('Now using smaller step-size.... ')

  return np.array(elbos),np.array(times), np.array(hp), variational_params, unpack_params




def padded_data(n_trials, n_neurons, max_len, spikes_n):
    '''
    Zero pads the raw spike data to be centre aligned and fixed length
    '''
    p_spikes_n = []
    for trial_i in range(n_trials):
        padded_s = np.zeros((n_neurons, max_len))
        if spikes_n[trial_i][1].size != max_len:
            diff = max_len - spikes_n[trial_i][1].size
            left_pad = math.ceil(diff/2)
            right_pad = math.floor(diff/2)
            for neuron_i in range(n_neurons):
                padded_s[neuron_i,:] = np.pad(spikes_n[trial_i][neuron_i], (left_pad, right_pad))
            p_spikes_n.append(padded_s)
        else:
            p_spikes_n.append(spikes_n[trial_i])
    
    return(p_spikes_n)

def stretched_data(n_trials, n_neurons, max_len, spikes_n):
    '''
    Stretches the raw spike data to be fixed length
    '''
    s_spikes_n = []
    for neuron_i in range(n_neurons):
        s_spike = np.zeros((n_trials,max_len))
        for trial_i in range(n_trials):
            factor = max_len/spikes_n[trial_i][neuron_i].size
            s_sig = librosa.core.resample(spikes_n[trial_i][neuron_i].astype(float), orig_sr=1, target_sr=1*factor)
            if s_sig.size != max_len:
                s_sig = s_sig[:-1].copy()
            s_spike[trial_i,:] = s_sig
        s_spikes_n.append(np.clip(np.rint(s_spike), 0, 100))
        
    return(s_spikes_n)

def pos_based_data(n_trials, n_neurons, max_len, spikes_n):
    '''
    Transforms the raw spike data to be based on position instead of time
    
    es_spikes_n can be used for the stretched down data to improve computation time
    but with slightly less accurate spike trains
    '''
    e_spikes_n = []    
    for neuron_i in range(n_neurons):
        e_spikes = np.zeros((n_trials, max_len*2))
        for trial_i in range(n_trials):
            even_spikes = traces.TimeSeries()
            for i in range(position_n[trial_i].size):
                even_spikes[position_n[trial_i][i]] = spikes_n[trial_i][neuron_i,i]
            sample_size = 160/(max_len*2)
            even_spikes = even_spikes.sample(sample_size)
            for i in range(len(even_spikes)):
                if i>(max_len*2-1):
                    break
                e_spikes[trial_i, i] = even_spikes[i][1]
        e_spikes_n.append(e_spikes)
        
    es_spikes_n = []
    for neuron_i in range(n_neurons):
        s_spike = np.zeros((n_trials,max_len))
        for trial_i in range(n_trials):
            factor = max_len/e_spikes_n[neuron_i][trial_i].size
            s_sig = librosa.core.resample(e_spikes_n[neuron_i][trial_i].astype(float), orig_sr=1, target_sr=1*factor)
            if s_sig.size != max_len:
                s_sig = s_sig[:-1].copy()
            s_spike[trial_i,:] = s_sig
        es_spikes_n.append(np.clip(np.rint(s_spike), 0, 100))
        
    return(es_spikes_n)

def warped_data(n_trials, n_neurons, max_len, fixed_spikes_n):
    '''
    Warps the padded/stretched spike data with shift/linear/piecewise linear warping
    '''
    f_spikes = np.dstack(fixed_spikes_n) #when using streched data
    #f_spikes = np.dstack(fixed_spikes_n).transpose(2,1,0) #when using padded data

    #model = ShiftWarping(smoothness_reg_scale=20.0)
    model = PiecewiseWarping(n_knots=0, warp_reg_scale=3.6e-6, smoothness_reg_scale=20.0)
    model.fit(f_spikes, iterations=184, warp_iterations=1001)

    all_neurons = np.arange(0, n_neurons)

    raw_spikes = SpikeData(*np.where(f_spikes), 0, f_spikes.shape[1]).select_neurons(all_neurons)
    est_align = model.transform(raw_spikes)

    b_spikes = est_align.bin_spikes(est_align.tmax)
    
    return(b_spikes)




print('Loading data ...... ')
#pth = 'Trimmed_data_0_orientation.npz' #Change this line! Specify path to data npz file.
#y_train = np.load(pth)['arr_0'][()]['y_train'] 
#N, nAL_neurons, D = AL_ex.shape
#tot_ex = np.concatenate([AL_ex, V1_ex], axis = 1).T


import math
import librosa
import traces
from affinewarp import PiecewiseWarping, ShiftWarping, SpikeData
import pickle
with open('ST263_Day1.pkl', 'rb') as f: #loading the data
    ST263_Day1 = pickle.load(f)
    
oasis = ST263_Day1['oasis']
position = ST263_Day1['position']
signals = ST263_Day1['signals']
spikes = ST263_Day1['spikes']
time_steps = ST263_Day1['time']
trial = ST263_Day1['trial']

oasis_n = []
position_n = []
signals_n = []
spikes_n = []
time_n = []

n_trials = trial[-1]+1
n_neurons = signals.shape[0]

for trial_i in range(n_trials): #separating the data by trial
    indeces = [i for i, value in enumerate(trial) if value == trial_i]

    oasis_n.append(oasis[:,indeces[0]:(indeces[-1]+1)])
    position_n.append(position[indeces[0]:(indeces[-1]+1)])
    signals_n.append(signals[:,indeces[0]:(indeces[-1]+1)])
    spikes_n.append(spikes[:,indeces[0]:(indeces[-1]+1)])
    time_n.append(time_steps[indeces[0]:(indeces[-1]+1)])
    

max_len = 0
for trial_i in range(n_trials): #determine the size of the trial of maximal length
    if signals_n[trial_i][0].size > max_len:
        max_len = signals_n[trial_i][0].size
    
p_data = padded_data(n_trials, n_neurons, max_len, spikes_n)
s_data = stretched_data(n_trials, n_neurons, max_len, spikes_n)
pos_data = pos_based_data(n_trials, n_neurons, max_len, spikes_n)
w_data = warped_data(n_trials, n_neurons, max_len, s_data) #change here between padded or stretched data (p_data or s_data)

#choose here which preprocessing method to follow

#y_train = np.dstack(p_data).transpose(2,0,1)
#y_train = np.dstack(s_data).transpose(0,2,1)
#y_train = np.dstack(pos_data).transpose(0,2,1)
y_train = w_data.transpose(0,2,1)


##################### Inference ######################
##set fourier params
minlens = 10 #assert a minimum scale for eigenvalue thresholding
condthresh = 1e8
nxc_ext = 0.1

##set variables to store optimization info
elbos = []
loads = []
recons = []


t = time.time()

#specify dimensionality
dim_noise = 6
dim_sig = 5


N_trs, tot_neurons, N = y_train.shape[0], y_train.shape[1],y_train.shape[2]

### convert to fourier
[By, wwnrm, Bffts, nxcirc] = gpf.comp_fourier.conv_fourier_mult_neuron(y_train, N, minlens,tot_neurons,nxcirc = np.array([N+nxc_ext*N]),condthresh = condthresh)
Bf = Bffts[0]
N_four = Bffts[0].shape[0]






####### Variational Inference (AUTOGRAD) ###############


tot_latents = N_trs*dim_noise + dim_sig # noise plus signal variational params
num_var_params = N_four*tot_latents # noise plus signal variational params
num_loadings_params = (dim_sig+1)*tot_neurons + dim_noise*tot_neurons ## add DC offset here


tee = time.time()
print('Starting optimization ...... ')
elbos,times, hps, variational_params,   unpack_params = optimize(y_train, num_var_params, tot_neurons, dim_sig,  dim_noise, N_four, N_trs)
print("elapsed time is ....", time.time() - tee)

lat_mean, lat_var, W_s,W_n, len_sc_sig, len_sc_noise = unpack_params(variational_params)



lat_mean_sig = np.reshape(lat_mean[:N_four*dim_sig], [dim_sig, N_four]) #reconstruct signal latents
lat_mean_noise = np.reshape(lat_mean[N_four*dim_sig:], [N_trs, dim_noise, N_four]) #reconstruct noise latents (trial by neuron by time)


recon_latent_time = lat_mean_sig@Bf #convert to time domain
recon_rates_sig= recon_latent_time.T@W_s[:dim_sig]
recon_noise_lat_time = lat_mean_noise@Bf[None, :] #number of trials by dim of noise

#Uncomment below to view rates per neuron
# recon_rates_sig = W[:dim_sig].T@recon_latent_time + W[dim_sig][:,None]
# recon_sum_lat = recon_rates_sig[None,:,:] + np.array([W_n[:dim_noise].T@recon_noise_lat_time[i,:,:]  for i in np.arange(N_trs)])

# recon_rates= softplus(recon_sum_lat)


########### Plotting #############
#Optimization of ELBO
plt.figure(1)
plt.plot(times, elbos)
plt.ylabel('ELBO')
plt.xlabel('Steps')

clock_cycle = time_n[0][1]-time_n[0][0] #for time-based data
#clock_cycle = 160/N #for pos-based data

plt.figure(2)
u, s, v = np.linalg.svd(recon_latent_time)
pclat = u.T@recon_latent_time
plt.plot(np.arange(0,N*clock_cycle,clock_cycle),pclat.T)
plt.ylabel('Signal PCs')
plt.xlabel('Time (s)')
plt.legend(['PC1','PC2','PC3','PC4','PC5'], loc="lower left", ncol = 3, fontsize=12)


plt.figure(3)
u, s, v = np.linalg.svd(recon_noise_lat_time[0,:,:])
pclat = u.T@recon_noise_lat_time[0,:,:]
plt.plot(np.arange(0,N*clock_cycle,clock_cycle),pclat[:4].T)
plt.ylabel('Noise PCs')
plt.xlabel('Time (s)')
plt.legend(['PC1','PC2','PC3','PC4'], fontsize=12)
#plt.ylim([-1,1])


## one trial
"""plt.figure(4)
u,s,v = np.linalg.svd(W_s[:-1], full_matrices = False) #remove DC offset
trial = 0
Y_n = W_n.T@recon_noise_lat_time[trial]
proj= v.T@v@Y_n
align_noise2sig = np.sum(np.square(proj), axis = 0)
orth_noise2sig = np.sum(np.square(Y_n - v.T@v@Y_n), axis = 0)
tot_var = np.sum(np.square(Y_n),axis = 0)
plt.plot(np.arange(0,N*clock_cycle,clock_cycle), align_noise2sig)
plt.plot(np.arange(0,N*clock_cycle,clock_cycle), orth_noise2sig)
plt.xlabel('time (s)')
plt.legend(['align','orth'])
plt.show()
plt.plot(np.arange(0,N*clock_cycle,clock_cycle), orth_noise2sig-align_noise2sig)
plt.plot(np.arange(0,N*clock_cycle,clock_cycle), np.clip(orth_noise2sig-align_noise2sig,0,1000))
plt.xlabel('time (s)')
plt.show()
plt.plot(np.arange(0,N*clock_cycle,clock_cycle), align_noise2sig/tot_var)
plt.plot(np.arange(0,N*clock_cycle,clock_cycle), orth_noise2sig/tot_var)
plt.xlabel('time (s)')
plt.legend(['align','orth'])"""


### summing over all trials
tot_align = 0
tot_orth = 0
tot_align_unnorm = 0
tot_orth_unnorm = 0
orth_filter = 0
start_trial = 0
end_trial = N_trs

for i in np.arange(start_trial,end_trial):
  u,s,v = np.linalg.svd(W_s[:-1], full_matrices = False)

  #Y_s = W.T@recon_lat_time_sixdims
  Y_n = W_n.T@recon_noise_lat_time[i,:,:]
  proj= v.T@v@Y_n
  align_noise2sig = np.sum(np.square(proj), axis = 0)
  orth_noise2sig = np.sum(np.square(Y_n - v.T@v@Y_n), axis = 0)

  orth_filter += orth_noise2sig-align_noise2sig

  tot_var = np.sum(np.square(Y_n),axis = 0)
  tot_align += align_noise2sig/tot_var
  tot_orth += orth_noise2sig/tot_var
  tot_align_unnorm += align_noise2sig
  tot_orth_unnorm += orth_noise2sig

plt.figure(5)
plt.plot(np.arange(0,N*clock_cycle,clock_cycle),tot_align/(end_trial-start_trial))
plt.plot(np.arange(0,N*clock_cycle,clock_cycle),tot_orth/(end_trial-start_trial))
plt.ylabel('L2 Norm (frac)')
plt.xlabel('Time (s)')
plt.legend(['align','orth'], loc="upper right", fontsize=12)
plt.show()
plt.plot(np.arange(0,N*clock_cycle,clock_cycle),tot_align_unnorm/(end_trial-start_trial))
plt.plot(np.arange(0,N*clock_cycle,clock_cycle),tot_orth_unnorm/(end_trial-start_trial))
plt.ylabel('L2 Norm')
plt.xlabel('Time (s)')
plt.legend(['align','orth'], loc="upper right", fontsize=12)
plt.show()
plt.plot(np.arange(0,N*clock_cycle,clock_cycle),orth_filter/(end_trial-start_trial))
plt.plot(np.arange(0,N*clock_cycle,clock_cycle),np.zeros(N))
plt.ylabel('L2 Norm diff (orth filter)')
plt.xlabel('Time (s)')


### summing over all trials
start_trial = np.arange(0,N_trs-10,10)
end_trial = np.arange(10,N_trs,10)

for j in range(len(start_trial)):
    tot_align = 0
    tot_orth = 0
    tot_align_unnorm = 0
    tot_orth_unnorm = 0
    orth_filter = 0
    
    for i in np.arange(start_trial[j],end_trial[j]):
        u,s,v = np.linalg.svd(W_s[:-1], full_matrices = False)

        #Y_s = W.T@recon_lat_time_sixdims
        Y_n = W_n.T@recon_noise_lat_time[i,:,:]
        proj= v.T@v@Y_n
        align_noise2sig = np.sum(np.square(proj), axis = 0)
        orth_noise2sig = np.sum(np.square(Y_n - v.T@v@Y_n), axis = 0)

        orth_filter += orth_noise2sig-align_noise2sig

        tot_var = np.sum(np.square(Y_n),axis = 0)
        tot_align += align_noise2sig/tot_var
        tot_orth += orth_noise2sig/tot_var
        tot_align_unnorm += align_noise2sig
        tot_orth_unnorm += orth_noise2sig

    plt.figure(6)
    plt.plot(np.arange(0,N*clock_cycle,clock_cycle),tot_align/(end_trial[j]-start_trial[j]))
    plt.plot(np.arange(0,N*clock_cycle,clock_cycle),tot_orth/(end_trial[j]-start_trial[j]))
    plt.ylabel('L2 Norm (frac)')
    plt.xlabel('Time (s)')
    plt.legend(['align','orth'], loc="upper right", fontsize=12)
    plt.show()
    plt.plot(np.arange(0,N*clock_cycle,clock_cycle),tot_align_unnorm/(end_trial[j]-start_trial[j]))
    plt.plot(np.arange(0,N*clock_cycle,clock_cycle),tot_orth_unnorm/(end_trial[j]-start_trial[j]))
    plt.ylabel('L2 Norm')
    plt.xlabel('Time (s)')
    plt.legend(['align','orth'], loc="upper right", fontsize=12)
    plt.show()
    plt.plot(np.arange(0,N*clock_cycle,clock_cycle),orth_filter/(end_trial[j]-start_trial[j]))
    plt.plot(np.arange(0,N*clock_cycle,clock_cycle),np.zeros(N))
    plt.ylabel('L2 Norm diff')
    plt.xlabel('Time (s)')
    plt.show()

