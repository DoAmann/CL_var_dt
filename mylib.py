import numpy as np
import matplotlib.pyplot as plt
import math 

import random as r

import time

import os
import re

import threading

class ResultThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._result = None
        self._exception = None

    def run(self):
        try:
            if self._target:
                self._result = self._target(*self._args, **self._kwargs)
        except Exception as e:
            self._exception = e

    def result(self):
        if self._exception:
            raise self._exception
        return self._result

#############
# COMPILING #
#############


def compile(on, imported_M, imported_mu, imported_g, imported_d, imported_S, L, N_Runs, offset, make_file, main_file, Par_file, card):
    seedline=91
    seedline2=92
    Par_fileline=27

    MUline		=14
    Gline		=15
    Mline		=16
    SIGMAline	=17
    Dline		=18
    Xline       =8
    Yline       =9

    if card=="rtx3090": arch="sm_86"
    if card=="a100" or card=="1g.10gb" or card=="A100": arch="sm_80"

    def modify_line_in_file(file_path, line_number, new_content):
        # Read all lines from the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Check if the line number is valid
        if line_number < 1 or line_number > len(lines):
            raise ValueError("Invalid line number")

        # Modify the specific line
        lines[line_number - 1] = new_content + '\n'

        # Write the modified lines back to the file
        with open(file_path, 'w') as file:
            file.writelines(lines)


    r.seed()

    modify_line_in_file(main_file,Par_fileline, "#include \""+Par_file+"\"")


    #modify_line_in_file(Par_file, MUline, "#define MU      "+"%0.5f" % (imported_mu)+"         //chemical potential")
    modify_line_in_file(Par_file, Gline, "#define G       "+"%0.5f" % (imported_g)
    +"         //coupling constant")
    modify_line_in_file(Par_file, Mline, "#define M       "+"%0.5f" % (imported_M)+"		        //dipole coupling (C_dd)" )
    modify_line_in_file(Par_file, Dline, "#define D       "+"%0.5f" % (imported_d)+"         //sigma of the z gaussian solution d^2=1/(m*omega_z)")
    
    modify_line_in_file(Par_file, Xline, "#define XSIZE "+ str(L))
    #modify_line_in_file(Par_file, Yline, "#define YSIZE "+ str(L))

    #tilting angle
    modify_line_in_file(Par_file, SIGMAline, "#define SIGMA   "+"%0.5f" % (imported_S)+"         //tilting angle")


    try: os.system("c++ -o Compute_costants Compute_costants.cpp")
    except: print("Error compling Compute_costants.cpp")
    print("./Compute_costants "+ Par_file)
    os.system("./Compute_costants "+ Par_file)

    
            

    N=0

    if on=="Helix":
        for i in range(N_Runs):

            modify_line_in_file(main_file,seedline, "unsigned long long seedvalue ="+str(int(r.random()*1e16))+"ULL;")
            modify_line_in_file(main_file,seedline2, "unsigned long long seedvalue2 ="+str(int(r.random()*1e16))+"ULL;")

            with open(make_file,"r") as f:
                lines=f.readlines()
            lines[0]="all: 2D_bosegascl"+str(N+offset)+"\n"
            lines[1]="2D_bosegascl"+str(N+offset)+": "+main_file+" "+Par_file+" Lattice.cu Propagator_without_cutoff.cu Observables_without_cutoff.cu SaveResults.cu"+"\n"
            lines[2]="	nvcc -std=c++17 -lstdc++fs -arch="+arch+" -Wno-deprecated-gpu-targets -lcurand -lcufft -o 2D_bosegascl"+str(N+offset)+" "+main_file+"\n"
            with open(make_file,"w") as f:
                f.writelines(lines)
            
            os.system("make -f "+make_file)
            print("Compiled "+ str(N+offset)+"\n", end='')

            with open("TableNamesPar","a") as save:
            	save.write(str(N)+"\t2D_bosegascl"+str(N+offset)+"\t"+str(imported_M)+"\t"+str(imported_mu)+"\t"+str(imported_g)+"\t"+str(imported_d)+"\t"+str(imported_S)+"\n")
            
            N+=1
    
    if on=="EINC":
        for i in range(N_Runs):

            #modify_line_in_file(main_file,seedline, "unsigned long long seedvalue ="+str(int(r.random()*1e16))+"ULL;")  
            #modify_line_in_file(main_file,seedline2, "unsigned long long seedvalue2 ="+str(int(r.random()*1e16))+"ULL;")        

            with open(make_file,"r") as f:
                lines=f.readlines()
            lines[0]="CUDA_LIB_DIR = /opt/spack_views/ido/targets/x86_64-linux/lib"+"\n"	
            lines[1]="LD_LIBRARY_PATH=$(CUDA_LIB_DIR)"+"\n"
            lines[2]="all: 2D_bosegascl"+str(N+offset)+"\n"
            lines[3]="2D_bosegascl"+str(N+offset)+": "+main_file+" "+Par_file+" Lattice.cu Propagator_without_cutoff.cu Observables_without_cutoff.cu SaveResults.cu"+"\n"
            lines[4]="\tnvcc -arch="+arch+" -Wno-deprecated-gpu-targets -L$(CUDA_LIB_DIR) --linker-options -rpath,$(LD_LIBRARY_PATH) -lcurand -lcufft -o  2D_bosegascl"+str(N+offset)+" "+main_file+"\n"	
            with open(make_file,"w") as f:
                f.writelines(lines)
            

            os.system("make -f "+make_file)
            print("Compiled "+ str(N+offset)+"\n", end='')

            with open("TableNamesPar","a") as save:
            	save.write(str(N)+"\t2D_bosegascl"+str(N+offset)+"\t"+str(imported_M)+"\t"+str(imported_mu)+"\t"+str(imported_g)+"\t"+str(imported_d)+"\t"+str(imported_S)+"\n")
            
            N+=1




######################
# ERROR SLURM CHECK  #
######################

def extract_str(input_string):
    pattern = r'_(\d+)\.out'
    match = re.search(pattern, input_string)
    if match:
        return match.group(1)
    else:
        return None

def find_error_in_all_slurm_files(directory, old_dir, Run_command, relaunch=True):
    list_of_errors = []
    for root, dirs, files in os.walk(directory):
            for file in files:
                if ".out" in file:
                    if "2d-dipolar_" in file:
                        try:
                            #print(file)
                            with open(directory+file, 'r') as f:
                                for line in f:
                                    condition = ("Error" in line) or ("error" in line) or ("Kill" in line ) or ("kill" in line)  or ("Stop" in line) or ("stop" in line)
                                    
                                    if condition:
                                        number=extract_str(file)
                                        print(f"Error found in file: "+file+"\n", end='')
                                        list_of_errors.append(file)
                                        if relaunch:
                                            job_id = os.popen("sbatch --array="+number+" jobscript.sh").read()
                                            watch(directory, job_id[-8:-1], old_dir, 1)
                                            os.system("mv "+file+" ../Error_Trash")
                                        break
                        except Exception as e:
                            print(f"Could not read file {file}: {e}"+"\n")
    return list_of_errors
                
def check_dir_creation_in_slurm_file(file_path, dir_path, old_dir):
    print("Checkig for errors in "+file_path+"\n", end='')
    time.sleep(10)
    #print(file_path, type(file_path))
    try:
        with open(file_path, 'r') as f:
            number=extract_str(file_path)
            string_to_look= "Directory created successfully:"
            if string_to_look not in f.readline():
                print(f"Error found in file: "+file_path+"\n")
                job_id = os.popen("sbatch --array="+number+" jobscript.sh").read()
                watch(dir_path, job_id[-8:-1], old_dir, 1)
                os.system("mv "+file_path+" ../Error_Trash")

    except Exception as e:
        print(f"Could not read file {file_path}: {e}"+"\n")

'''
def find_error_in_slurm_file(file_path, error_code, dir_path, old_dir, Run_command):
    print("Checkig for errors in "+file_path+"\n", end='')
    time.sleep(10)
    #print(file_path, type(file_path))
    try:
        with open(file_path, 'r') as f:
            number=extract_str(file_path)
            for line in f:
                if f"Error code {error_code}" in line:
                    print(f"Error {error_code} found in file: "+file_path+"\n")
                    job_id = os.popen("sbatch --array="+number+" jobscript.sh").read()
                    watch(dir_path, job_id[-8:-1], old_dir, 1)
                    os.system("mv "+file_path+" ../Error_Trash")
                    break
    except Exception as e:
        print(f"Could not read file {file_path}: {e}"+"\n")
'''

####################
# EVENT MONITORING #
####################

def counts_file_slurm(directory, m, M):
    count=0
    for filename in os.listdir(directory):
        if ".out" in filename:
            if "2d-dipolar_" in filename:
                number=extract_str(filename)
                if int(number) in range(m, M):
                    count+=1
    return count

def counts_finished(directory, m, M, verbose=True):
    count=[]
    for root, dirs, files in os.walk(directory):
            for file in files:
                if ".out" in file:
                    if "2d-dipolar_" in file:
                        #file_path = os.path.join(root, file)
                        try:
                            with open(directory+file, 'r') as f:
                                for line in f:
                                    if f"Finished simulation " in line:
                                        number=extract_str(file)
                                        if int(number) in range(m, M+1):
                                            count.append(file)
                                            if verbose: print(f"{file} is Finished, counter=", len(count), "\n", end='')
                                        break
                        except Exception as e:
                            print(f"Could not read file {file}: {e}"+"\n")
    return count



def watch(dir_path, job_id, old, N):
    count=0
    print("Monitoring directory:"+dir_path+"\n", end='')
    while(True):
        time.sleep(5)
        tmp = []
        for filename in os.listdir(dir_path): 
            if job_id in filename: 
                tmp.append(filename)
        
        for t in tmp:
            if not(t in old):
                thread = threading.Thread(target=check_dir_creation_in_slurm_file, args=(t, dir_path, tmp))
                thread.start()
                count+=1
        
        old = tmp
        if count==N: 
            print("End monitoring directory:"+dir_path+"\n", end='')
            break

def Finished(directory, offset, N_runs):
    count=0
    count = len(counts_finished(directory, offset , offset+N_runs-1))


    if count < N_runs: return False
    if count == N_runs: return True


#########
# Rho_s #
#########
class NanError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


def create_figure():
    fig= plt.figure()
    ax = fig.add_axes([0.13,0.13,0.84,0.85])
    
    ax.set_xlabel(r"$\vartheta/(10^4 a_\mathrm{s}^{-3})$")
    ax.set_ylabel(r"$\rho_\mathrm{s}/\rho_\mathrm{s}^\mathrm{c}$")

    ax.axhline(1 , color = "black", linestyle = "--",zorder=-1000)
    ax.axhline(0 , color = "black", linestyle = "--",zorder=-1000)
    #ax.axis(xmin=0.48,xmax=3.05)

    return fig, ax

def are_dicts_equal_except_key(dict1, dict2, key_to_ignore):
    # Create copies of both dictionaries without the key to ignore
    dict1_copy = dict1.copy()
    dict2_copy = dict2.copy()

    # Remove the specified key from both dictionaries
    dict1_copy.pop(key_to_ignore, None)
    dict2_copy.pop(key_to_ignore, None)

    # Compare the remaining parts of both dictionaries
    return dict1_copy == dict2_copy


def compute_rhos(step_number, path, LAT, T, mu, g, M, D, SIGMA, cutted_percentage, runs, excluded=[], update_plot=True, fig=0, ax=0):
    

    def rhos(p2, N_tot, LAT, T):
        return (1-p2/N_tot/T)*N_tot/LAT**2/(T/np.pi)

    def rhos_err_band(p2, N_tot, LAT, T, err):
        return (1-(p2+err)/N_tot/T)*N_tot/LAT**2/(T/np.pi)
    
    def rhos_err(p2_err, N_tot_err, LAT, T):
        return (np.pi/T/LAT**2) * np.sqrt((N_tot_err**2 + p2_err**2/T**2))

    #start = time.time()

    if M==0: M=int(M)
    if g==0: g=int(g)
    if D==0: D=int(D)

    folder=path+"DATA/BoseCL_Simulation_T_"+str(T)+"_mu_"+str(mu)+"_g_"+str(g)+"_M_"+str(M)+"_SIGMA_"+str(SIGMA)+"_D_"+str(D)+"_OMEGAX_0_OMEGAY_0_OMEGAZ_0_components_1_lattice_"+str(LAT)+"_"+str(LAT)+"_1_32/Run"

    #folder=path+"BoseCL_Simulation_T_"+str(T)+"_mu_"+str(mu)+"_g_"+str(g)+"_M_"+str(M)+"_SIGMA_"+str(SIGMA)+"_D_"+str(D)+"_OMEGAX_0_OMEGAY_0_OMEGAZ_0_components_1_lattice_"+str(LAT)+"_"+str(LAT)+"_1_32/Run"

    evol=np.array([])
    evol_err=np.array([])


    parameters = {}

    # Open the file and read line by line
    with open(folder+"1/Parameters.txt", 'r') as file:
        for line in file:
            key, value = line.strip().split('\t')  # Split at the space
            parameters[key] = value  # Add the key-value pair to the dictionary

    
    p2s = np.array([np.loadtxt(folder+str(1)+"/Scalars.txt", usecols=1)])
    Ns = np.array([np.loadtxt(folder+str(1)+"/Scalars.txt", usecols=0)])
        
    for r in range(2, runs+1):

        tmp_parameters = {}

        # Open the file and read line by line
        with open(folder+str(r)+"/Parameters.txt", 'r') as file:
            for line in file:
                key, value = line.strip().split('\t')  # Split at the space
                tmp_parameters[key] = value  # Add the key-value pair to the dictionary
        
        if not are_dicts_equal_except_key( parameters, tmp_parameters, 'seed'):
            print("Error Run"+str(r)+" has different parameters.")
            exit()

        
        tmp_P2 = np.loadtxt(folder+str(r)+"/Scalars.txt", usecols=1)
        tmp_N = np.loadtxt(folder+str(r)+"/Scalars.txt", usecols=0)
        if len(tmp_P2) != len(p2s[0]) or len(tmp_N) != len(p2s[0]): 
            print("Error Run"+str(r)+" has different steps lenght.")
            exit()
        p2s = np.append(p2s, [tmp_P2], axis=0)
        Ns = np.append(Ns, [tmp_N], axis=0)

    
    i_0 = int(cutted_percentage * len(p2s[0]))

    for i in range(1, len(p2s[0]) - i_0):
        var=np.array([])
        for r in range(0, runs):

            if r+1 in excluded: continue

            p2_last = p2s[r][i_0+i]
            p2_t0 = p2s[r][i_0]          
            
            if (np.isnan(p2_last)): 
                message = "Error: P2=nan at line "+ str(i_0 + i) +"\n"
                raise NanError(message)
                
            # i -> i+1 because it take care of the fact that we save a step 0 and step 1, 
            # than the step and the index are not in phase
            var=np.append(var,(i_0 + (i+ 1))/((i+1))*(p2_last - (i_0) / (i_0 + (i+1)) * p2_t0))
        
        evol=np.append(evol,np.mean(var))    
        evol_err=np.append(evol_err,np.var(var)**0.5/(runs - len(excluded))**0.5) 



    evol2=np.array([])
    evol2_err=np.array([])
    
    for i in range(1, len(p2s[0]) - i_0):
        var=np.array([])
        for r in range(0, runs):
            
            if r+1 in excluded: continue
            
            N_last = Ns[r][i_0 + i]
            N_t0 = Ns[r][i_0]

            if (np.isnan(N_last)): 
                message="Error: N=nan at line "+ str(i_0 + i) +"\n"
                raise NanError(message)

            # i -> i+1 because it take care of the fact that we save a step 0 and step 1, 
            # than the step and the index are not in phase
            var=np.append(var,(i_0 + (i+ 1))/((i+1))*(N_last- (i_0) / (i_0 + (i+1)) *N_t0))     
     
        evol2=np.append(evol2,np.mean(var))    
        evol2_err=np.append(evol2_err,np.var(var)**0.5/(runs - len(excluded))**0.5) 


    #print(time.time()-start)
    #print( rhos(evol[-1], evol2[-1], LAT, T), rhos_err(evol_err[-1], evol2_err[-1], LAT, T ))
    
    '''
    #OLD# 

    start = time.time()
    folder=path+"BoseCL_Simulation_T_"+str(T)+"_mu_"+str(mu)+"_g_"+str(g)+"_M_"+str(M)+"_SIGMA_"+str(SIGMA)+"_D_"+str(D)+"_OMEGAX_0_OMEGAY_0_OMEGAZ_0_components_1_lattice_"+str(LAT)+"_"+str(LAT)+"_1_32/Run"

    evol=np.array([])
    evol_err=np.array([])

    for i,t in enumerate(times):

        var=np.array([])
        for r in range(1, runs+1):

            if r in excluded: continue
            
            p2_last = np.loadtxt(folder+str(r)+"/Scalars_"+str(t)+".txt",skiprows=1,usecols=(3))
            p2_t0 = np.loadtxt(folder+str(r)+"/Scalars_"+str(t0)+".txt",skiprows=1,usecols=(3))

            #if i == 10: print(p2_t0, p2_last)

            if (np.isnan(p2_last)): 
                message = "Error: P2=nan in "+ folder+str(r)+"/Scalars_"+str(t)+".txt"+"\n"
                raise NanError(message)
                
            var=np.append(var,t/(t-t0)*(p2_last-t0/t*p2_t0))
        evol=np.append(evol,np.mean(var))    
        evol_err=np.append(evol_err,np.var(var)**0.5/(runs - len(excluded))**0.5) 
    
            
    evol2=np.array([])
    evol2_err=np.array([])
    
    for t in times:
        var=np.array([])
        for r in range(1, runs+1):
            
            if r in excluded: continue
            
            N_last = np.loadtxt(folder+str(r)+"/Scalars_"+str(t)+".txt",skiprows=0,max_rows=1,usecols=(3))
            N_t0 = np.loadtxt(folder+str(r)+"/Scalars_"+str(t0)+".txt",skiprows=0,max_rows=1,usecols=(3))

            if (np.isnan(N_last)): 
                message="Error: N=nan in "+ folder+str(r)+"/Scalars_"+str(t)+".txt"+"\n"
                raise NanError(message)

            var=np.append(var,t/(t-t0)*(N_last-t0/t*N_t0))    
        evol2=np.append(evol2,np.mean(var))    
        evol2_err=np.append(evol2_err,np.var(var)**0.5/(runs - len(excluded))**0.5)   
    
   
    #print(time.time()-start)
    #'''

    #print(evol2[-1]/LAT**2,evol2_err[-1]/LAT**2,(1-evol[-1]/evol2[-1]/1.25)*evol2[-1]/LAT**2/(1.25/np.pi),(evol_err[-1]/evol2[-1]/1.25)*evol2[-1]/LAT**2/(1.25/np.pi))
    
    if update_plot:
        dt=float(parameters['dt'])
        cutted_snapshot = (len(p2s[0])-1-i_0)
        number_of_steps = (len(p2s[0])-2 )*int(parameters['SNAPSHOT_TRIGGER'])
        if number_of_steps != int(parameters['steps']):
            print('Error: number_of_steps != parameters[\'steps\']')
            exit()
        theta = np.linspace(    i_0*int(parameters['SNAPSHOT_TRIGGER'])*dt, 
                                int(number_of_steps)*dt, 
                                cutted_snapshot)
        theta /= 10**4 # to have a nicer plot
        ax.plot(theta, rhos(evol, evol2, LAT, T), label=str(mu))
        ax.fill_between(theta ,rhos_err_band(evol, evol2, LAT, T, evol_err), rhos_err_band(evol, evol2, LAT, T, -evol_err),alpha=0.2)
        ax.legend()
        fig.savefig("tmp_rhos_"+str(step_number)+".png")
    
    return rhos(evol[-1], evol2[-1], LAT, T), rhos_err(evol_err[-1], evol2_err[-1], LAT, T )



def step(on, i, dir_path, LAT, T, mu, eps_dd, g_2d, S, t0_percentage, N_Runs, offset, make, main, par, card, Run_command, plot_args):
        
    h_l =  (1/math.sqrt(mu))
    d = h_l
    g = (d * math.sqrt(2*math.pi)) * g_2d
    M = 3*g*eps_dd


    mu = round(mu ,5)
    M = round(M ,5)
    d = round(d ,5)
    g = round(g ,5)
    S = round(S, 5)

    compile(on, M, mu, g, d, S, LAT, N_Runs, offset, make, main, par, card)
    print("Compilation ended, mu ="+str(mu)+"\n", end='')
    old = os.listdir(dir_path)
    job_id = os.popen(Run_command+str(offset)+"-"+str(offset+N_Runs-1)+" jobscript.sh").read()
    watch(dir_path, job_id[-8:-1], old, N_Runs)

    
    #for i in range(offset, offset+N_Runs):
    #   os.system(Run_command+str(i)+" > 2d-dipolar_"+str(i)+".out 2>&1 &")
    #    #watch(dir_path, "2d-dipolar_"+str(i)+".out", old, N_Runs, Run_command)   
    
    
    while not(Finished(dir_path, offset, N_Runs)):
        time.sleep(60)
        find_error_in_all_slurm_files(dir_path, old, Run_command)
    
    print("Finished mu = "+ str(mu)+ "\n", end='')
        
    return compute_rhos(i, dir_path, LAT, T, mu, g, M, d, S, t0_percentage, N_Runs, [], *plot_args)
    

##########
# New mu #
##########

class JError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class GuessError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


def new_mu(rhos_, mu, i):
    
    rhos = rhos_ -1
    j = i
    while True:
        new = (rhos[i+1]*mu[j]-rhos[j]*mu[i+1])/(rhos[i+1]-rhos[j])
        
    
        if new >= 10*np.max(mu) and new <= 0:  
            j -= 1
            if j < 0:
                if i==0:
                    m="Bad guess. \n"                 
                    raise GuessError(m)    
                else:
                    m= "j too small. Secant method failed"+"\n"
                    raise JError(m)
      
        
        else: return new

def check_to_stop(rho, err):
    if rho-1 < err and rho-1 > -err: return True
    else: return False




#################
# Secant Method #
#################

def change_guess(r, g, d):
    if r[0] >1 and r[1]>1:
        return 0.5*g



def Secant_method_function(on, home, LAT, T, g_2d, eps_dd, SIGMA, guess, delta, t0_percentage, N_Runs, offset, make_file, main_file, Par_file, card, Run_command, plot=True):
    
    #copy
    make_file_copy = make_file+"_copy"
    main_file_copy = main_file.split('.', 1)[0].strip()+"_copy.cu"
    Par_file_copy  = Par_file.split('.', 1)[0].strip()+"_copy.h"


    os.system("cp "+home+make_file+" "+home+make_file_copy)
    os.system("cp "+home+main_file+" "+home+main_file_copy)
    os.system("cp "+home+Par_file +" "+home+Par_file_copy)
    
    time.sleep(1)
    
    fig, ax=create_figure()
    plot_args= [plot, fig, ax]

    mu_0 = round((1+delta)*guess,5)
    mu_1 = round((1-delta)*guess,5)

    mu_s = np.array([mu_0, mu_1])
    #mu_s[0] = 0.125    
    
    flag = True
    i = 0
    while(flag):            

        if (i==0): 
            try:

                thread_0 = ResultThread(target=step, args=(on, 0, home, LAT, T, mu_s[0], eps_dd, g_2d, SIGMA,t0_percentage, N_Runs, offset, make_file, main_file, Par_file, card, Run_command, plot_args))
                thread_0.start()
                        
                offset+=N_Runs 

                
                thread_1 = ResultThread(target=step, args=(on, 1, home, LAT, T, mu_s[1], eps_dd, g_2d, SIGMA, t0_percentage, N_Runs, offset,  make_file_copy, main_file_copy,   Par_file_copy, card, Run_command, plot_args))
                thread_1.start()

                thread_0.join()
                thread_1.join()


                rho_0, rho_0_err = thread_0.result()
                rho, rho_err = thread_1.result()
            
            
            except NanError as e:
                print(f"Cannot compute rhos: {e.message}")
                break
            
            offset+=N_Runs 

            rhos = np.array([rho_0, rho])
            rhos_err =  np.array([rho_0_err, rho_err])

            if check_to_stop(rho_0, rho_0_err):
                print("EUREKA: it's "+ str(mu_s[0])+"\n")
                flag = False

            if check_to_stop(rho, rho_err):
                print("EUREKA: it's "+ str(mu_s[1])+"\n")
                flag = False
            
            try:
                mu_s=np.append(mu_s, round(new_mu(rhos, mu_s, i=0), 5))
            except GuessError as e:
                print(f"Cannot compute new_mu: {e.message}")
                guess, delta= change_guess(rhos, guess, delta)
                
                mu_0 = round((1+delta)*guess,5)
                mu_1 = round((1-delta)*guess,5)

                mu_s = np.array([mu_0, mu_1])
                continue


            i+=1


        
        else:
            try:
                rho, rho_err = step(on, i, home, LAT, T, mu_s[i], eps_dd, g_2d, SIGMA, t0_percentage, N_Runs, offset, make_file, main_file, Par_file, card, Run_command, plot_args)
            except NanError as e:
                print(f"Cannot compute rhos: {e.message}")
                break
            
            rhos= np.append(rhos, rho)
            rhos_err= np.append(rhos_err, rho_err)

            offset+=N_Runs 
            try:
                mu_s=np.append(mu_s, new_mu(rhos, mu_s, i-1))
            except JError as e:
                print(f"Cannot compute new_mu: {e.message}")
                break

        
        if check_to_stop(rhos[-1], rhos_err[-1]):
            print("EUREKA"+"\n")
            flag = False

        if i==15:
            print("Stopped. Too many runs."+"\n")
            break
        
        i+=1