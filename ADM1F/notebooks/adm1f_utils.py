import sys, glob, os, os.path
import subprocess
import numpy as np
import pandas as pd
import xlrd
import time
import lhsmdu    #generate Latin Hypercube samples
from scipy.stats import gaussian_kde as kde

def check_filename(fname):
    '''
    checks file existence
    '''
    fcheck = os.path.isfile(fname)
    if not fcheck:
        print ('File does not exist',fname)
    return fcheck

def get_output_names():    
    '''
    This function reads the excel file 'out_sludge.xls'
    and outputs the index and name of the output values
    Input: 
        None
    Output: 
        output_unit: unit
        output_name: output parameter name
    '''
    
    # Grab the names of all the outputs
    fname='../notebooks/out_sludge.xls'
    check_filename(fname)
    wb = xlrd.open_workbook(fname)

    output_name = []
    output_unit = []
    sh = wb.sheet_by_index(1)     #output sheet
    for i in range(67):
        output_name.append(sh.cell(1,i).value)
        output_unit.append(sh.cell(2,i).value)
    
    return (output_name,output_unit)

def get_output_header(): # check if we still use it
    # Grab the names of all the outputs
    output_dict = {}
    wb = xlrd.open_workbook('../notebooks/out_sludge.xls')
    sh = wb.sheet_by_index(1)     #sheet
    for i in range(67):
        cell_value = sh.cell(1,i).value
        output_dict[i] = cell_value

    key_list = list(output_dict.keys()) # index_out
    val_list = list(output_dict.values()) # name_out
    
    return val_list

def get_param_names():    
    '''
    This function reads the excel file 'out_sludge.xls'
    and outputs the index and name of the param values
    Input: 
        None
    Output: 
        param_index: index 
        param_name : output parameter name
    '''

    # Grab the names of all the parameters
    fname='../notebooks/out_sludge.xls'
    check_filename(fname)
    wb = xlrd.open_workbook(fname)
    
    param_name = []
    param_index = []
    sh = wb.sheet_by_index(0)     #output sheet
    for i in range(100):   
        param_name.append(sh.cell(0,i).value)
        param_index.append(sh.cell(1,i).value)
        
    return (param_name,param_index)

def get_influent_names():
    '''
    This function reads the excel file 'out_sludge.xls'
    and outputs the index and name of the influent values
    Input: 
        None
    Output: 
        influent_index: index 
        influent_name : output parameter name
    '''
    influent_dict = {}
    fname='../notebooks/out_sludge.xls'
    wb = xlrd.open_workbook(fname)
    sh = wb.sheet_by_index(2)     #sheet
    for i in range(28):
        cell_value = sh.cell(1,i).value
        influent_dict[i] = cell_value

    influent_index = list(influent_dict.keys())
    influent_name = list(influent_dict.values())
    
    return (influent_name,influent_index)

def create_a_sample_matrix(variable='influent',method='uniform',percent=0.1,sample_size=100):
    '''
    This function will create a sampling file (var_'variable'.csv)
    Inputs:
        variable: 'params', 'influent', 'ic' 
        method: 'lhs', 'uniform' 
        percent: 0.1 
        sample_size: 100
    Outputs:
        index_factor : indexes of the variables participated in sampling
    '''
    
    # 1.Load input and compute the interval of possible changes for nonzero values
    # factor: influent, parameter, ic
    factor = np.loadtxt(variable + '.dat')
    val_factor = [i for i in factor]
    
    if variable == 'params':
        index_remove = [54, 60, 77, 78, 79, 85, 86, 93]
        index_factor = [i for i in range(len(val_factor)) if i not in index_remove]
    elif variable == 'influent':
        index_factor = [i for i in range(len(val_factor)) if val_factor[i]!=0]
    else: #'ic'
        index_remove = [7, 9, 10, 24, 25, 32, 33, 37, 38, 39, 40, 41, 42]   #7, 32, 42 is for being too small
        index_factor = [i for i in range(len(val_factor)) if i not in index_remove]  
    
    val_factor_low = [round(val_factor[i]*(1-percent),5) for i in index_factor]
    val_factor_high = [round(val_factor[i]*(1+percent),5) for i in index_factor]
    val_factor_diff = [round(val_factor_high[i]-val_factor_low[i],5) for i in range(len(index_factor))]
    # or (val_factor_high[i]+val_factor_low[i])/2  
    
    # 2. Choose sampling method: lhs or uniform
    if method == 'lhs':
        # Latin Hypercube Sampling, each column indicates a sample point
        # This is like an initialization, almost return the same results
        l = lhsmdu.sample(len(index_factor),sample_size)    #type:matrix
        # Latin Hypercube Sampling of factor(influent or parameter or ic)
        l = lhsmdu.resample().T        # Latin Hypercube Sampling from uniform, after transpose, each row indicates a sample point
        mat_factor_diff = np.diag(val_factor_diff)
        sample_factor = l*mat_factor_diff + val_factor_low      # Scaling to Latin Hypercube Sampling of influent
    elif method == 'uniform':
        ## Uniform Sampling ##
        l = np.random.uniform(size=(sample_size, len(index_factor)))
        sample_factor = l*val_factor_diff + val_factor_low

    np.savetxt('var_%s.csv'%(variable),sample_factor,delimiter=',',fmt='%13.5f')
    print ('Saves a sampling matrix [sample_size,array_size] into var_%s.csv'%(variable))
    print ('sample_size,array_size: ',sample_factor.shape)
    print ('Each column of the matrix corresponds to a variable perturbed 100 times around its original value ')
    print ('var_%s.csv'%(variable), 'SAVED!')
    
    return index_factor
    
def adm1f_output_sampling(ADM1F_EXE,variable,index):
    '''
    This function will run the ADM1F_EXE using a sampling file (var_'variable'.csv) produced in "create_a_sample"
    Inputs:
        ADM1F_EXE : path to adm1f executable
        variable: 'params', 'influent', 'ic' 
        index: indices of the nonzero sample values 

    Outputs:
        execution time
    '''

    var_samples = pd.read_csv('var_%s.csv'%(variable),header=None) 
    data=np.loadtxt(variable+'.dat')
    [sample_size,array_size]=var_samples.shape
    #sample_size=#len(index) # number of samples 
    output = np.zeros((sample_size,67))
    exec_time = np.zeros(sample_size)
    
    #index=index_factor
    file_line='-'+variable+'_file'
    # loops over all samples, changes current vairable file, and save outputs
    for i in range(sample_size):
        if i in [66,94] and variable=='params':        ###[66,94]for params
            continue

        for j in range(len(index)):
            data[index[j]] = var_samples[j][i]
        
        #print (i,index[i])
        filename=variable+'_cur.dat'
        np.savetxt(filename, data) 
        # run the adm1f with an updated input file and store data from the last output file
        # adm1f will runs with the specified configuration
        # e.g.: ADM1F_EXE -steady -params_file params_cur.dat
        command_line = '$ADM1F_EXE' + ' -steady'
        command_line = command_line + ' ' + file_line + ' ' + filename
        print (command_line)
        
        start_time = time.time()
        status=subprocess.call(command_line, shell=True)
        #status=subprocess.check_call(command_line)
        exec_time[i]=time.time() - start_time
        #print("--- %s seconds ---" % exec_time)
        if status:
            print('ADM1F failed to execute...')
        outputfile=getlastoutput()
        #save last output file into the ith row of variable output
        output[i,:]=np.loadtxt(outputfile, skiprows=2, unpack=True)
        removeoldoutputs()

    np.savetxt('outputs_%s.csv'%(variable), output, delimiter=',')
    print ('All %s runs were successfully computed'%(sample_size))
    print ('outputs_%s.csv'%(variable), 'SAVED!')
    
    return exec_time
    
def getlastoutput():
    '''
    Returns the last indicator filename
    '''
    filelist = os.listdir(os.getcwd())
    filelist = filter(lambda x: not os.path.isdir(x), filelist)
    names=[s for s in filelist if 'indicator' in s]
    idx_max = max([int(s.split('-')[1].split('.')[0]) for s in names])
    lastname = "indicator-{:03d}.out".format(idx_max)
    return lastname

def removeoldoutputs():
    filelist = glob.glob(os.path.join("*.out"))    
    for f in filelist:
        os.remove(f)
        
def get_param_header():
    # Grab the names of all the params
    par_dict = {}
    wb = xlrd.open_workbook('../docs/jupyter_notebook/out_sludge.xls')
    sh2 = wb.sheet_by_index(0)     #sheet
    for i in range(100):
        cell_value = sh2.cell(0,i).value
        par_dict[i] = cell_value # par name
        #print(cell_value)
        
    val_list = list(par_dict.values())
    
    return val_list

def reactor1(opt='', Q=100, Vliq=300, t_resx=30):
    '''
    One-phase reactor. 
    Input parameters:
        opt:     additional option specification
        Q:       flux 
        Vliq:    volume of the liquid 
        t_resx:  retention time
    Output:
        val_out: an array of outputs correspoding to last time of the simulation 
    '''
    #Construct the commnad line 
    command_line = '$ADM1F_EXE ' + opt 
    #Read the influent data, change the Q, and save the data in influent_cur.dat
    data=np.loadtxt('influent.dat')
    data[26] = Q #index 26 in the influent corresponds to the flux
    np.savetxt('influent_cur.dat', data, fmt='%5.6f')    
    command_line = command_line + ' -Vliq '+str(Vliq)
    command_line = command_line + ' -t_resx '+str(t_resx) 
    command_line = command_line + ' -influent_file' + ' influent_cur.dat'
    print('Reactor run, phase-one:')
    print(command_line)
    
    #execute the ADM1F SRT
    subprocess.call(command_line, shell=True)

    #NOTE: check_all will produce error
    #status=subprocess.check_call(command_line)
    
    last_file=getlastoutput()
    print (last_file)
    val_out = np.loadtxt(last_file,skiprows=2, unpack=True)
    removeoldoutputs()

    return val_out

def reactor2(**kwargs):
    '''
    Two phase reactor, Q, Vliq and t_resx for phase 1 and 2 can be set to certain values
    Returns output of both phase 1 and phase 2
    (Only need to put values in the function when it is not using the default)
    Vliq1: Vliq of phase 1
    Vliq2: Vliq of phase 2
    t_resx1: t_resx of phase 1
    t_resx2: t_resx of phase 2
    Q1: flow rate of phase 1
    Q2: flow rate of phase 2
    '''
    ################## Phase I ##################
    phase1_out2 = reactor1(Q=kwargs['Q1'], Vliq=kwargs['Vliq1'], t_resx=kwargs['t_resx1'])
    
    ################## Phase II ##################
    command2 = '$ADM1F_EXE'
    arg1 = {}
    data=np.loadtxt('influent_cur.dat')
    ## Soluable component
    #Indexes for solids (particulates and biomass) and soluble contents
    X_idx = [i for i in range(12,24)] # index for the particulate part (SRT)
    S_idx = [i for i in range(26) if i not in X_idx]  # index for the soluble part (HRT)
    for j in S_idx:
        if j == 9:  
            data[j] = phase1_out2[j]/12000  # unit of C
        elif j == 10:
            data[j] = phase1_out2[j]/14000  # unit of N
        else:
            data[j] = phase1_out2[j]/1000   # /1000 because of the unit
    ## Particulate component
    for j in X_idx:
        data[j] = 0.17*phase1_out2[j]/1000   #0.17 is the membrane efficiency
    ## Q: flow rate
    if 'Vliq1' in kwargs.keys():
        arg1['V'] = kwargs['Vliq1']
    if 't_resx1' in kwargs.keys():
        arg1['t_resx'] = kwargs['t_resx1']
    a = afun(data[26],0.17,**arg1)
    data[26] = a*data[26]/(1+a)
    ## set parameters for the phase-two run
    if 'Q2' in kwargs.keys():
        data[26] = kwargs['Q2']
    np.savetxt('influent_cur.dat', data, fmt='%5.6f')
    if 'Vliq2' in kwargs.keys():
        command2 = command2 + ' -Vliq '+str(kwargs['Vliq2'])
    if 't_resx2' in kwargs.keys():
        command2 = command2 + ' -t_resx '+str(kwargs['t_resx2']) 
    command2 = command2 + ' -influent_file' + ' influent_cur.dat'
    print('Reactor run, phase-two:')
    print(command2)
    
    #execute the ADM1F SRT
    subprocess.call(command2, shell=True)
    #NOTE: check_all will produce error
    #status=subprocess.check_call(command_line)
    
    last_file=getlastoutput()
    print (last_file)
    phase2_out = np.loadtxt(last_file,skiprows=2, unpack=True)
    #removeoldoutputs()
 
    return phase1_out2, phase2_out

def reactor2_cat(init_sample,verbose = 'on',**kwargs):
    '''
    Two phase reactor, we will update the cation values for phase 1 and 2 to control PH.
    Q, Vliq and t_resx for phase 1 and 2 can be set to certain values
    Will create output file storing the phase 1 result and phase 2 result
    (Only need to put values in the function when it is not using the default)
    target_1: target for phase 1
    target_2: target for phase 2
    Vliq1: Vliq of phase 1
    Vliq2: Vliq of phase 2
    t_resx1: t_resx of phase 1
    t_resx2: t_resx of phase 2
    Q1: flow rate of phase 1
    Q2: flow rate of phase 2
    '''
    ################## Phase I ##################
    command1, command2 = '$ADM1F_EXE -steady', '$ADM1F_EXE -steady'
    infl_path = 'influent.dat'
    params_path = 'params.dat'
    ic_path = 'ic.dat'
    X_idx = [i for i in range(12,24)]
    S_idx = [i for i in range(26) if i not in X_idx]  
    print('verbose:',verbose)
    
    arg1, arg2 = {}, {}
    if 'Q1' in kwargs.keys():
        arg1["Q"] = kwargs['Q1']
    if 'Vliq1' in kwargs.keys():
        arg1["Vliq"] = kwargs["Vliq1"]
        command1 = command1 + ' -Vliq '+str(kwargs['Vliq1'])
    if 't_resx1' in kwargs.keys():
        arg1["t_resx"] = kwargs["t_resx1"]
        command1 = command1 + ' -t_resx '+str(kwargs['t_resx1']) 
    if 'target_1' in kwargs.keys():
        cat_val1 = ph_control(kwargs['target_1'],init_sample,1,infl_path,params_path,ic_path,verbose=verbose,**arg1)[0]
        data=np.loadtxt('influent_cur.dat')
        data[24] = cat_val1 
        np.savetxt('influent_cur.dat', data, fmt='%5.6f')
        command1 = command1 + ' -influent_file' + ' influent_cur.dat'
        print(command1)
        subprocess.call(command1, shell=True)   
        phase1_out2 = np.loadtxt(getlastoutput(),skiprows=2, unpack=True)
        print('Phase 1, after changing cation to {:.6f} kmole/m3, new PH = {}'.format(cat_val1, phase1_out2[26]))
    else:
        command1 = command1 + ' -influent_file' + ' influent_cur.dat'
        print(command1)
        subprocess.call(command1, shell=True)
        phase1_out2 = np.loadtxt(getlastoutput(),skiprows=2, unpack=True)
        print('Phase 1, without changing cation, predicted PH = {}'.format(phase1_out2[26]))
    removeoldoutputs()
    
    ################## Phase II ##################
    data=np.loadtxt('influent_cur.dat')
    ## Soluable component
    for j in S_idx:
        if j == 9:  
            data[j] = phase1_out2[j]/12000  # unit of C
        elif j == 10:
            data[j] = phase1_out2[j]/14000  # unit of N
        else:
            data[j] = phase1_out2[j]/1000   # /1000 because of the unit
    ## Particulate component
    for j in X_idx:
        data[j] = 0.17*phase1_out2[j]/1000   #0.17 is the membrane efficiency
    ## Q: flow rate
    arga = {}
    if 'Vliq1' in kwargs.keys():
        arga['V'] = kwargs['Vliq1']
    if 't_resx1' in kwargs.keys():
        arga['t_resx'] = kwargs['t_resx1']
    a = afun(data[26],0.17,**arga)
    data[26] = a*data[26]/(1+a)
    ##--
    np.savetxt('influent_cur.dat', data, fmt='%5.6f')
    if 'Q2' in kwargs.keys():
        arg2["Q"] = kwargs['Q2']
    if 'Vliq2' in kwargs.keys():
        arg2["Vliq"] = kwargs["Vliq2"]
        command2 = command2 + ' -Vliq '+str(kwargs['Vliq2'])
    if 't_resx2' in kwargs.keys():
        arg2["t_resx"] = kwargs["t_resx2"]
        command2 = command2 + ' -t_resx '+str(kwargs['t_resx2']) 
    if 'target_2' in kwargs.keys():
        cat_val2 = ph_control(kwargs["target_2"],init_sample,1,infl_path,params_path,ic_path,verbose=verbose,**arg2)[0]   #[0] is to convert one-item list to a float
        if isinstance(cat_val2, float):  #this means cation needs to be adjusted
            data=np.loadtxt('influent_cur.dat')
            data[24] = cat_val2
            np.savetxt('influent_cur.dat', data, fmt='%5.6f')
            command2 = command2 + ' -influent_file' + ' influent_cur.dat'
            print('')
            print(command2)
            subprocess.call(command2, shell=True)
            phase2_out = np.loadtxt(getlastoutput(),skiprows=2, unpack=True)
            print('Phase 2, after changing cation to {:.6f} kmole/m3, new PH = {}'.format(cat_val2, phase2_out[26]))
    else:
        command2 = command2 + ' -influent_file' + ' influent_cur.dat'
        print('')
        print(command2)
        subprocess.call(command2, shell=True)
        phase2_out = np.loadtxt(getlastoutput(),skiprows=2, unpack=True)
        print('Phase 2, without changing cation, predicted PH = {}'.format(phase2_out[26]))
    removeoldoutputs()

    return

def afun(Q0,eff,**kwargs):
    '''
    Sets up filtration effiency of the dynamic membrane
    Inputs:
        eff   : percentage of solids pass thorugh the dynamic membrane (membrane efficiency)
        Q0    : flow rate of the effluent from the first-phase reactor
        kwargs: is a list of 'V' volume, and 't_resx' SRT
    Outputs:
        nu: amount of solids retained by the dynamic membrane per unit time (kg COD/d)
        de: amount of solids passed through the dynamic membrane per unit time (kg COD/d)
    '''
    V = 3400
    t_resx = 0
    if 'V' in kwargs.keys():
        V = kwargs['V']
    if 't_resx' in kwargs.keys():
        t_resx = kwargs['t_resx']
    nu = Q0*(1-eff)                
    de = V/(V/Q0+t_resx) - eff*Q0  
    
    return nu/de-1

def ph(cat, verbose = 'on', **kwargs):
    '''
    Given cation value, returns the predicted PH and the cation value
    Inputs:
        cat (float): cation value
        kwargs: put Vliq or t_resx value if needed
        verbose: (on/off) prints messages 
    Outputs:
        cat (float): cation value
        ph_out: correspoding to the cation ph values    
    '''
    
    #Construct the commnad line 
    command_line = '$ADM1F_EXE -steady'
    #Read the influent data, change the Q, and save the data in influent_cur.dat
    data=np.loadtxt('influent.dat')
    data[24] = cat #change cation value in influent.dat
    np.savetxt('influent_cur.dat', data, fmt='%5.6f')    

    command_line = command_line + ' -influent_file' + ' influent_cur.dat'
    if 'Q' in kwargs.keys():
        data=np.loadtxt('influent_cur.dat')
        data[26] = kwargs['Q']
        np.savetxt('influent_cur.dat', data, fmt='%5.6f') 
    if 'Vliq' in kwargs.keys():
        command_line = command_line + ' -Vliq '+str(kwargs['Vliq'])
    if 't_resx' in kwargs.keys():
        command_line = command_line + ' -t_resx '+str(kwargs['t_resx'])
    
    if verbose == 'on':
        print('Reactor run, ph phase-one:')
        print(command_line)
    
    #execute the ADM1F SRT
    subprocess.call(command_line, shell=True)

    #NOTE: check_all will produce error
    #status=subprocess.check_call(command_line)
    
    last_file=getlastoutput()
    with open(last_file) as f:
        lines = f.read().splitlines()
    if lines[28] == 'nan.':
        ph_out = -9999
    else:
        ph_out = float(lines[28])
    
    removeoldoutputs()
    
    return ph_out, cat

def ph_control(target,init_sample,num_cat,infl,params,ic,threshold=8,verbose = 'on',**kwargs):
    '''
    Calculates several most possible cation values that will lead to the target PH
    Inputs:
        target (class object): target object with target PH value and certain standard variation
        init_sample: the inital sample used to update the needed cation value
        num_cat (int): number of cation values that will be returned
        infl, params, ic (string): path of each input file
        kwargs: put Vliq or t_resx value if needed
        verbose: (on/off) prints messages 
    Outputs:
        val_return: set of possible cation values 
    '''
    # load the data from the default files
    vinfl= np.loadtxt(infl)
    vparams = np.loadtxt(params)
    vic = np.loadtxt(ic)
    # save the data into the current files 
    np.savetxt('influent_cur.dat', vinfl)
    np.savetxt('params_cur.dat', vparams)
    np.savetxt('ic_cur.dat', vic)
    
    cat0 = np.loadtxt('influent_cur.dat')[24]
    predict0 = ph(cat0,verbose=verbose,**kwargs)[0]   #the PH when no actions are taken
    
    print('Predicted PH is', predict0)
    
    if (predict0 < target.ph - 3*target.sig):
        pfinit_sample = []
        for i in init_sample:
            pfinit_sample.append(ph(i,verbose=verbose,**kwargs)[0])
            
        idx = [i for i in range(len(init_sample)) if pfinit_sample[i]==-9999]
        init_sample_update = [init_sample[i] for i in range(len(init_sample)) if i not in idx]
        pfinit_sample_update = [pfinit_sample[i] for i in range(len(init_sample)) if i not in idx]

        pdf_pfinit = kde(pfinit_sample_update)

        r = target.pdf(pfinit_sample_update)/pdf_pfinit(pfinit_sample_update)
        M = max(r)
        unif = np.random.uniform(0,1,len(pfinit_sample_update))
        up_idx = [i for i in range(len(init_sample_update)) if r[i]/M>unif[i]]
        up_sample = [init_sample_update[i] for i in up_idx]
        up_pf_sample = [pfinit_sample_update[i] for i in up_idx]

        lst = [abs(i-target.sig) for i in up_pf_sample]
        cat_iidx = sorted((x,i) for (i,x) in enumerate(lst))[:num_cat]
        cat_idx = [y for (x,y) in cat_iidx]
        val_return = [up_sample[i] for i in cat_idx]
    elif (predict0 > threshold):
        raise ValueError("PH is above the threshold "+str(threshold)+" even no actions are taken")
    else:
        val_return = "No action is needed."
    
    return val_return