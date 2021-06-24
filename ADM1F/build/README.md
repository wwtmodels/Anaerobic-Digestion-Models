## Compile ADM1F

1.  ADM1F uses external numerical library package PETSc. First download PETSc:

    `$ cd build; git clone -b release https://gitlab.com/petsc/petsc.git petsc`
    
    `$ cd petsc; git checkout v3.14`

2.  Set **PETSC_DIR** and **PETSC_ARCH** in your environmental variables. We suggest to put these lines in your `~/.bashrc` or similar files (`~/.bash_profile` on Mac OS X). Once you add it into the bash file, run `source  ~/.bash_profile`:

    `$ export PETSC_DIR=/path-to-my-ADM1F-folder/build/petsc`
    
    and
    
    `$ export PETSC_ARCH=macx-debug`
    
    **Make sure** that ‘adolc-utils’ folder is in the ‘build' folder. 

3.  Configure PETSC:

    `$ ./configure --download-mpich --with-cc=clang --with-fc=gfortran --with-debugging=0 --download-adolc PETSC_ARCH=macx-debug --with-cxx-dialect=C++11 --download-colpack`

**NOTE**: that these are for Mac OSX. If you are installing on a linux machine, then replace **clang** with **gcc**. Also, sometimes turning off `--with-fc=0` could help with compilation. This step will take awhile.

4.  If configuration goes well, you can then compile. This step will take awhile too.:

    `$ make PETSC_DIR=/path-to-my-ADM1F-folder/build/petsc PETSC_ARCH=macx-debug all`

5.  After compilation, PETSc will show you how to test your installation (testing is optional).

6.  Navigate back to the `build` folder (`cd ../`) and compile adm1f:

    `$ make adm1f`
    
    or
    
    `$ make`

7.  Set **ADM1F_EXE** in your environmental variable. Add this line in your `~/.bashrc` or similar files (`~/.bash_profile` on Mac OS X).  Once you add it into the bash file, do not forget to `source  ~/.bash_profile`:
     
    `$ export ADM1F_EXE=path-to-my-ADM1F-folder/build/adm1f`

8.  **NOTE**: There are two versions of the ADMF1: the original version  (adm1f.cxx), and the modified version of the model (`build/adm1f_srt.cxx`, see [ADM1F_SRT](https://elchin.github.io/ADM1F_docs/compile.html#adm1f-srt)). 
    

## Running ADM1F

1. Make sure that **ADM1F_EXE:** is not empty (see step 7 from the previous section).:

    `$ echo $ADM1F_EXE`

2. Navigate to the `simulations` folder and run the model:

    `$ $ADM1F_EXE`
    
    or using command-line options (see 4 and 5):
    
    `$ $ADM1F_EXE -ts_monitor -steady`

3. Note that adm1f will look for three files `ic.dat`, `params.dat`, and `influent.dat`, which contain the initial conditions (45 values), parameters (100 values), and influent values (28 values), see [here](https://elchin.github.io/ADM1F_docs/inouts.html).

4. The command-line options are:

    *  -Cat [val] - mass of Cat+ added [kmol/m3]
    *  -Vliq [val] - volume of liquid [m3]
    *  -Vgas [val] - volume of liquid [m3]
    *  -t_resx [val] -SRT adjustment: t_resx = SRT-HRT, [d] (works only for adm1f_srt.cxx)
    *  -params_file [filename] - specify params filename (default is params.dat)
    *  -ic_file [filename] - specify initial conditions filename (default is ic.dat)
    *  -influent_file [filename] - specify influent filename (default is influent.dat)
    *  -ts_monitor - shows the timestep and time information on screen
    *  -steady - run as steady state else runs as transient
    *  -debug - gives out more details on the screen

5. More command-line options can be found [here](<https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetFromOptions.html>).
6. Additional details on the ADM1F_SRT can be found [here](https://elchin.github.io/ADM1F_docs/compile.html#adm1f-srt).
