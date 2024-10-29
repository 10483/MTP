import numpy as np
import time
import json


def alpha2F(om): #quadratic approximation for the phonon spectrum
    return om**2

def pairbreakingdistributions(E,om,Delta): #distribution for the pair breaking interaction
    omminE = om-E
    dist =  1/(E**2-Delta**2)**0.5   *(E*(omminE)+Delta**2)/((omminE)**2-Delta**2)**0.5
    return dist


def phonon_emiss_dist(om,E,spectrum,Delta): #distribution for the phonon emission interaction
    dist = spectrum(om)*(E-om)/((E-om)**2-Delta**2)**0.5   *(1-Delta**2/(E*(E-om)))
    return dist



#DISCRETIZATION
#These fucntions calculate the possible outcomes of all interactions and corresponding chances.
#The outcomes are indexes in the energy-array, which used as an input to define what energies are possible in the simulation

def descretize_init_phon(spectrum,energy): #the initial phonon spectrum

    om_allowed = energy
    outcomes = np.arange(len(om_allowed))
    chances = spectrum(om_allowed)/sum(spectrum(om_allowed))
    print("Finished initial phonon descretisation")
    return outcomes, chances

def descretize_pair_breaking(energy,stepsize,Delta): #the pair breaking interaction
    om_allowed  =energy
    outcomes = []
    chances = []
    loc_delta = round(1/stepsize)
    indexarray = np.arange(0,len(energy)+1,1)
    for i in range(len(om_allowed)):
        indexes = indexarray[loc_delta:i-loc_delta+1] #define the allowed range of the outcomes
        outcomes.append(indexes)
        E = energy[indexes]
        chance = pairbreakingdistributions(E,om_allowed[i],Delta) #calculate the value of the probability distribution
        if len(chance) > 1:
            chance[-1] = pairbreakingdistributions(E[-1]-stepsize/4,om_allowed[i],Delta)/2 #the last bin needs to be halved in order to prevent an infinite chance, see report
        norm_chance = chance/sum(chance) #normalise the chances
        chances.append(norm_chance)
    print("finished pair breaking descretisation")
    return outcomes, chances

def descretize_phonon_emiss(spectrum,energy,stepsize,Delta):
    E_allowed = energy

    outcomes = []
    chances = []

    indexarray = np.arange(0,len(energy)+1,1)
    loc_delta = round(1/stepsize)

    for i in range(len(E_allowed)):
        indexes = indexarray[0:(i+1)-loc_delta] #define the allowed range of the outcomes
        outcomes.append(indexes)

        om = energy[indexes]
        chance = phonon_emiss_dist(om,E_allowed[i],spectrum,Delta) #calculate the value of the probability distribution
        if len(chance) > 1:
                chance[-1] = phonon_emiss_dist(om[-1]-stepsize/4,E_allowed[i],spectrum,Delta)/2  #the last bin needs to be halved in order to prevent an infinite chance, see report
        norm_chance = chance/sum(chance) #normalise the chances
        chances.append(norm_chance)

    print("finished phonon emission descretisation")
    return outcomes, chances

def pick_init_phon(Q,init_phon,energy,stepsize): #function to initialise the first group of phonons with energy equal to Q, the photon energy
    outcomes,chances = init_phon
    phonons = np.zeros(len(energy))

    while sum(phonons*energy) < Q:
        index = np.random.choice(outcomes,p=chances)
        phonons[index] += 1
    phonons[index] += -1
    totalenergy = np.sum(phonons*energy)

    lastenergy = int(Q-totalenergy)
    lastindex = int(lastenergy/stepsize)
    phonons[lastindex] += 1
    return phonons


def pick_quasis(index,pairbreaking,energy):
    outcomes, chances = pairbreaking
    try:
        out = outcomes[index]
        probs = chances[index]

        energy1 = np.random.choice(out,p=probs)
        energy2 = index-energy1
    except:
        print(energy[index])
        print(energy[outcomes[index]])
        print(outcomes[index],chances[index])
    quasis  = np.array([energy1,energy2])

    diff = energy[energy1]+energy[energy2]-energy[index]
    if np.abs(diff) > 1/10:
        print("no conservation:", diff,'Delta deviation')
        print(index,energy1,energy2)

    return quasis.astype(int)


def pick_phonon(index,emission,energy):
    outcomes, chances = emission[0], emission[1]
    try:
        phonon = np.random.choice(outcomes[index],p=chances[index])
    except:
        print(energy[index])
        print(outcomes[index],chances[index])
    return phonon.astype(int)


#main simulation loop
def simulation(Q,energy,cycles,stepsize,pairbreaking,emission,init_phon,control):
    start_time = time.time()
    N = np.zeros(cycles)
    ph = np.zeros(cycles)
    efficiency = np.zeros(cycles)
    E_mean = np.zeros(cycles)
    om_mean = np.zeros(cycles)

    L = len(energy)
    qlimit = round(3/stepsize) #limits where energies of phonons and quasiparticle cannot cause the creation of more quasiparticles
    plimit = round(2/stepsize)
    pairbreaking[1][plimit] = [1]
    #at a phonon energy 2*Delta, the discretisation does not function properly due to the fact that there is just one outcome (two Delta quasparticles)
    #possible and this results in infinite probability, which is why we need to set in manually

    for i in range(cycles):
        #some text to make the interface more clear
        print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~") #to separate iterations
        print("Cycle number: {}".format(i+1))
        #initialise the first phonons and an empty list for the quasiparticles
        quasiparticles = np.zeros(L)
        phonons = np.zeros(L)
        initphonons = (pick_init_phon(Q,init_phon,energy,stepsize))
        phonons += initphonons

        #cycle that repeats until all interacting particles are exhausted
        while sum(phonons[plimit:])>0 or sum(quasiparticles[qlimit:])>0:

            for p in range(plimit,L):
                if phonons[p] > 0:
                    for j in range(int(phonons[p])):
                        phonons[p] += -1
                        indexes = pick_quasis(p,pairbreaking,energy)
                        quasiparticles[indexes] += 1

            for q in range(qlimit,L):
                if quasiparticles[q] > 0:
                    for j in range(int(quasiparticles[q])):
                        #print(quasiparticles[q])
                        quasiparticles[q] += -1
                        index = pick_phonon(q,emission,energy)
                        phonons[index] += 1
                        newquasis = q - index
                        quasiparticles[newquasis] += 1



        #the sum of the quasiparticle array gives the no. of quasiparticles
        N[i]= sum(quasiparticles)
        ph[i] = sum(phonons)
        efficiency[i]= (sum(quasiparticles)*(1+stepsize/2)/Q)
        E_mean[i] = sum(quasiparticles*energy)/sum(quasiparticles)
        om_mean[i] = sum(phonons*energy)/sum(phonons)

        energycontrol = sum(phonons*energy)+sum(quasiparticles*energy)
        stop_time = time.time()
        Delta_t = stop_time-start_time
        print("Time passed: {} s".format(round(Delta_t,2)))
        if control == True: #print control variables if True
            print("Efficiency: {} ".format(efficiency[i]))
            print("Number of quasiparticles: {}".format(N[i]))
            print("Total energy: {} \u0394".format(round(energycontrol),3))
            print("Phonon energy: {} \u0394".format(round(sum(phonons*energy))))
            print("Quasiparticle energy: {} \u0394".format(round(sum(quasiparticles*energy))))
            print("Mean phonon energy: {} \u0394".format(round(om_mean[i],2)))
            print("Mean quasiparticle energy: {} \u0394".format(round(E_mean[i],2)))
            print("Energy deficiency: {} \u0394".format(sum(initphonons*energy)-energycontrol))

    return N, ph, efficiency, E_mean,om_mean

#SAVING DATA
def savedata(data,name,spec,cycles,wd):
    filename = name+str(spec)+"_"+"wd"+str(int(wd))+"_"+"2024-1-3"+"_"+str(cycles)+"its"
    data = list(data)
    with open("Data\\"+filename, 'w') as file:
        json.dump(data, file)
