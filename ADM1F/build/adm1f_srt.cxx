
static char help[] = "ADM1 hydrolysis model using PETSc. Solves the entire DAE system including pH and Sh2 calculations.\n";

/*F
    Uses ADOLC for automatic differentiation for Jacobian calculation of the implicit solve of the DAE. For steady-state, Pseudo time-stepping is used.

     Helpful runtime monitoring options:
         -ts_view                  -  prints information about the solver being used
         -ts_monitor               -  prints the progess of the solver
         -ts_adapt_monitor         -  prints the progress of the time-step adaptor
         -ts_monitor_lg_timestep   -  plots the size of each timestep (at each time-step)
         -ts_monitor_lg_solution   -  plots each component of the solution as a function of time (at each timestep)
         -ts_monitor_lg_error      -  plots each component of the error in the solution as a function of time (at each timestep)
         -draw_pause -2            -  hold the plots a the end of the solution process, enter a mouse press in each window to end the process

         -ts_monitor_lg_timestep -1  -  plots the size of each timestep (at the end of the solution process)
         -ts_monitor_lg_solution -1  -  plots each component of the solution as a function of time (at the end of the solution process)
         -ts_monitor_lg_error -1     -  plots each component of the error in the solution as a function of time (at the end of the solution process)
         -lg_use_markers false       -  do NOT show the data points on the plots
         -draw_save                  -  save the timestep and solution plot as a .Gif image file

F*/

#include <petscts.h>
#include "adolc-utils/drivers.cxx"
#include <adolc/adolc.h>

#define MAXLINE 1000

typedef struct {
  Vec         initialconditions;
  Vec         params;
  Vec         interface_params;
  Vec         influent;
  Vec         adm1_output;
  Vec         asm1_output;
  Vec         indicator;
  PetscScalar V[2]; // Store the volumes
  AdolcCtx    *adctx; /* Automatic differentiation support */
  PetscScalar rwork;
  PetscBool   debug;
  PetscScalar Cat_mass; // store the Cation mass addition to maintain pH
  PetscBool   set_Cat_mass;
  PetscScalar t_resx;
  PetscBool set_t_resx;
} AppCtx;


PetscErrorCode ReadParams(AppCtx *ctx, char *filename)
{
  FILE           *fp;
  PetscErrorCode ierr;
  char           line[MAXLINE];
  PetscInt       i;
  PetscInt       num_params = 100;
  PetscScalar    *param, val;

  PetscFunctionBegin;

  ierr = VecCreate(PETSC_COMM_WORLD,&ctx->params);CHKERRQ(ierr); // single processor only
  ierr = VecSetSizes(ctx->params,PETSC_DECIDE,num_params);CHKERRQ(ierr);
  ierr = VecSetFromOptions(ctx->params);CHKERRQ(ierr);

  fp = fopen(filename,"r");
  ierr = PetscPrintf(MPI_COMM_SELF,"Reading parameters in file: %s\n",
                     filename);CHKERRQ(ierr);
  /* Check for valid file */
  if (!fp) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_FILE_OPEN,"Can't open  "
                    "file: %s",filename);

  ierr = VecGetArray(ctx->params,&param);CHKERRQ(ierr);
  for (i = 0; i < num_params; i++) {
      fgets(line,MAXLINE,fp);
      sscanf(line,"%lf",&val);
      param[i] = val;
  }
  ierr = VecRestoreArray(ctx->params,&param);CHKERRQ(ierr);
  if (ctx->debug) {
    ierr = VecView(ctx->params,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  fclose(fp);
  PetscFunctionReturn(0);
}

PetscErrorCode ReadInfluent(AppCtx *ctx, char *filename)
{
  FILE           *fp;
  PetscErrorCode ierr;
  char           line[MAXLINE];
  PetscInt       i;
  PetscInt       num_influent = 28;
  PetscScalar    *influent, val;

  PetscFunctionBegin;

  ierr = VecCreate(PETSC_COMM_WORLD,&ctx->influent);CHKERRQ(ierr); // single processor only
  ierr = VecSetSizes(ctx->influent,PETSC_DECIDE,num_influent);CHKERRQ(ierr);
  ierr = VecSetFromOptions(ctx->influent);CHKERRQ(ierr);

  fp = fopen(filename,"r");
  ierr = PetscPrintf(MPI_COMM_SELF,"Reading influent values in file: %s\n",
                     filename);CHKERRQ(ierr);
  /* Check for valid file */
  if (!fp) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_FILE_OPEN,"Can't open  "
                    "file: %s",filename);

  ierr = VecGetArray(ctx->influent,&influent);CHKERRQ(ierr);
  for (i = 0; i < num_influent; i++) {
      fgets(line,MAXLINE,fp);
      sscanf(line,"%lf",&val);
      influent[i] = val;
  }
  ierr = VecRestoreArray(ctx->influent,&influent);CHKERRQ(ierr);
  if (ctx->debug) {
    ierr = VecView(ctx->influent,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  fclose(fp);
  PetscFunctionReturn(0);
}

PetscErrorCode ReadInitialConditions(AppCtx *ctx, char *filename)
{
  FILE           *fp;
  PetscErrorCode ierr;
  char           line[MAXLINE];
  PetscInt       i;
  PetscInt       num_ic = 45;
  PetscScalar    *ic, val;

  PetscFunctionBegin;

  ierr = VecCreate(PETSC_COMM_WORLD,&ctx->initialconditions);CHKERRQ(ierr);// single processor only
  ierr = VecSetSizes(ctx->initialconditions,PETSC_DECIDE,num_ic);CHKERRQ(ierr);
  ierr = VecSetFromOptions(ctx->initialconditions);CHKERRQ(ierr);

  fp = fopen(filename,"r");
  ierr = PetscPrintf(MPI_COMM_SELF,"Reading initial condition values in file: %s\n",
                     filename);CHKERRQ(ierr);
  /* Check for valid file */
  if (!fp) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_FILE_OPEN,"Can't open  "
                    "file: %s",filename);

  ierr = VecGetArray(ctx->initialconditions,&ic);CHKERRQ(ierr);
  for (i = 0; i < num_ic; i++) {
      fgets(line,MAXLINE,fp);
      sscanf(line,"%lf",&val);
      ic[i] = val;
  }
  ierr = VecRestoreArray(ctx->initialconditions,&ic);CHKERRQ(ierr);
  if (ctx->debug) {
    ierr = VecView(ctx->initialconditions,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  fclose(fp);
  PetscFunctionReturn(0);
}


PetscErrorCode DigestParToInterfacePar(AppCtx *ctx)
{
  PetscErrorCode ierr;
  PetscInt            num_interface_params = 19;
  const PetscScalar   *params; 
  PetscScalar         *interface_params;
  PetscInt            i;

  PetscFunctionBegin;


  ierr = VecCreate(PETSC_COMM_WORLD,&ctx->interface_params);CHKERRQ(ierr); // single processor only
  ierr = VecSetSizes(ctx->interface_params,PETSC_DECIDE,num_interface_params);CHKERRQ(ierr);
  ierr = VecSetFromOptions(ctx->interface_params);CHKERRQ(ierr);

  ierr = VecGetArrayRead(ctx->params,&params);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->interface_params,&interface_params);CHKERRQ(ierr);
  
  interface_params[0] = 0.0;
  interface_params[1] = 0.79;
  interface_params[2] = 0.0;
  interface_params[3] = params[7]*14;
  interface_params[4] = params[5]*14;
  interface_params[5] = params[22]*14;
  interface_params[6] = params[6]*14;
  interface_params[7] = params[6]*14;
  for (i = 8; i < 19; i++) 
  interface_params[i] = params[i+69];

  ierr = VecRestoreArray(ctx->interface_params,&interface_params);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(ctx->params,&params);CHKERRQ(ierr);

  if (ctx->debug) {
    ierr = PetscPrintf(MPI_COMM_SELF,"Interface Params:\n");CHKERRQ(ierr);
    ierr = VecView(ctx->interface_params,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);

}



PetscErrorCode ProcessADM1(Vec adm1_sol,AppCtx *ctx)
{ 
  /* Post-process on the ADM1 solution to get output in the standardized form */

  /* Output vector adm1_out entries are in this order:
  * y[0] : Ssu = monosaccharides (kg COD/m3)
  * y[1] : Saa = amino acids (kg COD/m3)
  * y[2] : Sfa = long chain fatty acids (LCFA) (kg COD/m3)
  * y[3] : Sva = total valerate (kg COD/m3)
  * y[4] : Sbu = total butyrate (kg COD/m3)
  * y[5] : Spro = total propionate (kg COD/m3)
  * y[6] : Sac = total acetate (kg COD/m3)
  * y[7] : Sh2 = hydrogen gas (kg COD/m3)
  * y[8] : Sch4 = methane gas (kg COD/m3)
  * y[9] : Sic = inorganic carbon (kmole C/m3)
  * y[10] : Sin = inorganic nitrogen (kmole N/m3)
  * y[11] : Si = soluble inerts (kg COD/m3)
  * y[12] : Xc = composites (kg COD/m3)
  * y[13] : Xch = carbohydrates (kg COD/m3)
  * y[14] : Xpr = proteins (kg COD/m3)
  * y[15] : Xli = lipids (kg COD/m3)
  * y[16] : Xsu = sugar degraders (kg COD/m3)
  * y[17] : Xaa = amino acid degraders (kg COD/m3)
  * y[18] : Xfa = LCFA degraders (kg COD/m3)
  * y[19] : Xc4 = valerate and butyrate degraders (kg COD/m3)
  * y[20] : Xpro = propionate degraders (kg COD/m3)
  * y[21] : Xac = acetate degraders (kg COD/m3)
  * y[22] : Xh2 = hydrogen degraders (kg COD/m3)
  * y[23] : Xi = particulate inerts (kg COD/m3)
  * y[24] : scat+ = cations (metallic ions, strong base) (kmole/m3)
  * y[25] : san- = anions (metallic ions, strong acid) (kmole/m3)
  * y[26] : flow rate (m3/d)
  * y[27] : temperature (deg C)
  * y[28:32] : dummy states
  * y[33] : pH = pH within AD system
  * y[34] : S_H+ = protons (kmole/m3)
  * y[35] : Sva- = valerate (kg COD/m3)
  * y[36] : Sbu- = butyrate (kg COD/m3)
  * y[37] : Spro- = propionate (kg COD/m3)
  * y[38] : Sac- = acetate (kg COD/m3)
  * y[39] : Shco3- = bicarbonate (kmole C/m3)
  * y[40] : Sco2 = carbon dioxide (kmole C/m3)
  * y[41] : Snh3 = ammonia (kmole N/m3)
  * y[42] : Snh4+ = ammonium (kmole N/m3)
  * y[43] : Sgas,h2 = hydrogen concentration in gas phase (kg COD/m3)
  * y[44] : Sgas,ch4 = methane concentration in gas phase (kg COD/m3)
  * y[45] : Sgas,co2 = carbon dioxide concentration in gas phase (kmole C/m3)
  * y[46] : pgas,h2 = partial pressure of hydrogen gas (bar)
  * y[47] : pgas,ch4 = partial pressure of methane gas (bar)
  * y[48] : pgas,co2 = partial pressure of carbon dioxide gas (bar)
  * y[49] : pgas,total = total head space pressure (H2+CO2+CH4+H2O) (bar)
  * y[50] : qgas = gas flow rate normalised to atmospheric pressure (m3/d)
  * y[51] : Sh2_in = hydrogen gas in influent  (m3/d)
  * y[52] : RT = retention time  (d)
  * y[53] : ACN = acetate capacity number (kg COD/m3/d)/(kg COD/m3/d)
  */
  PetscErrorCode    ierr;
  const PetscScalar *x, *params, *influent;
  PetscScalar       *y;
  PetscReal R, T_op, T_base, P_atm, p_gas_h2o, P_gas, k_P, q_gas, V_liq, p_gas_h2, p_gas_ch4, p_gas_co2, kLa, K_H_h2, K_H_ch4, K_H_co2, K_w, pK_w_base, factor;   
  PetscInt           i;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(ctx->params,&params);CHKERRQ(ierr);
  R = params[77];
  T_base = params[78];
  T_op = params[79]; /*in Kelvin */
  P_atm = params[93];
  V_liq = ctx->V[0];
  kLa = params[94];
  pK_w_base = params[80];
  k_P = params[99];
  factor = (1.0/T_base - 1.0/T_op)/(100.0*R);
  K_H_h2 = 1.0/pow(10,(-187.04/T_op+5.473))*55.6/1.01325;     /* atm*L/mol; modified van't Hoff-Arrhenius relationship; MetCalf & Eddy, p68; conversions 55.6 mole H2O/L; 1 atm = 1.01325 */
  K_H_ch4 = 1.0/pow(10,(-675.74/T_op+6.880))*55.6/1.01325; /* atm*L/mol; modified van't Hoff-Arrhenius relationship; MetCalf & Eddy, p68 */
  K_H_co2 = 1.0/pow(10,(-1012.40/T_op+6.606))*55.6/1.01325; /* atm*L/mol; modified van't Hoff-Arrhenius relationship; MetCalf & Eddy, p68 */
  K_w = pow(10,-pK_w_base)*exp(55700.0*factor); /* T adjustment for K_w */
  p_gas_h2o = pow(10, (5.20389-1733.926/(T_op-39.485)));  /* Antoine equation for water pressure for temp range 31C-60C */
  ierr = VecRestoreArrayRead(ctx->params,&params);CHKERRQ(ierr);


  ierr = VecGetArrayRead(adm1_sol,&x);CHKERRQ(ierr);

  ierr = VecGetArrayRead(ctx->influent,&influent);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->adm1_output,&y);CHKERRQ(ierr);

  for (i = 0; i < 26; i++) 
  y[i] = x[i]; /* Ssu, Saa, Sfa, Sva, Sbu, Spro, Sac, Sh2, Sch4, SIC, SIN, SI, Xxc, Xch, Xpr, Xli, Xsu, Xaa, Xfa, Xc4, Xpro, Xac, Xh2, XI, Scat, San */ 
  y[26] = influent[26];   		/* flow */
  y[27] = T_op - 273.15;  	/* Temp degC */
  y[28] = x[37];   /* Dummy state 1, soluble */
  y[29] = x[38];   /* Dummy state 2, soluble */   
  y[30] = x[39];   /* Dummy state 3, soluble */
  y[31] = x[40];   /* Dummy state 1, particulate */
  y[32] = x[41];   /* Dummy state 2, particulate */

  p_gas_h2 = x[32]*R*T_op/16.0;
  p_gas_ch4 = x[33]*R*T_op/64.0;
  p_gas_co2 = x[34]*R*T_op;
  P_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o;

  q_gas = k_P*(P_gas - P_atm);
  /*if (q_gas < 0)
    q_gas = 0.0;*/

  y[33] = -log10(x[42]);    	/* pH; x[42]=H+ */
  y[34] = x[42];            	/* SH+ */  
  y[35] = x[26];  	          /* Sva- */
  y[36] = x[27];  				    /* Sbu- */
  y[37] = x[28];   			      /* Spro- */
  y[38] = x[29];   		       	/* Sac- */
  y[39] = x[30];   		      	/* SHCO3- */
  y[40] = x[43]; 			        /* SCO2 */
  y[41] = x[31];  		    		/* SNH3 */
  y[42] = x[44];      	  		/* SNH4+; SIN - SNH3  */
  y[43] = x[32];             	/* Sgas,h2 */
  y[44] = x[33];             	/* Sgas,ch4 */
  y[45] = x[34];             	/* Sgas,co2 */
  y[46] = p_gas_h2;
  y[47] = p_gas_ch4;
  y[48] = p_gas_co2;
  y[49] = P_gas;                /* total head space pressure from H2, CH4, CO2 and H2O */
  y[50] = q_gas*P_gas/P_atm; 	  /* The output gas flow is recalculated to atmospheric pressure (normalization) */
  y[51] = x[7];                /* S_h2_in */
  y[52] = V_liq/(influent[26]);		    /* Retention Time (days) = V/Q; for steadystate definition*/
  y[53] = ctx->rwork;     			      /* acetate capacity number (ACN) in (kg COD/m3/d)/(kg COD/m3/d) */


  ierr = VecRestoreArrayRead(ctx->influent,&influent);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(adm1_sol,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->adm1_output,&y);CHKERRQ(ierr);

  PetscFunctionReturn(0);

}


PetscErrorCode ProcessADM1toASM1(AppCtx *ctx)
{
  /* Converts ADM1 output to ASM1 output */
  /* y[0] stop signal is not set */
  /* asm1_output[0] = y[i-1] for all i > 0 */

  /*
  * Output vector:
  * y[0] : stop signal =  nonzero value stops the simulation using the STOP block
  * separated by Demux
  * y[1] : Ssu = monosaccharides (kg COD/m3)
  * y[2] : Saa = amino acids (kg COD/m3)
  * y[3] : Sfa = long chain fatty acids (LCFA) (kg COD/m3)
  * y[4] : Sva = total valerate (kg COD/m3)
  * y[5] : Sbu = total butyrate (kg COD/m3)
  * y[6] : Spro = total propionate (kg COD/m3)
  * y[7] : Sac = total acetate (kg COD/m3)
  * y[8] : Sh2 = hydrogen gas (kg COD/m3)
  * y[9] : Sch4 = methane gas (kg COD/m3)
  * y[10] : Sic = inorganic carbon (kmole C/m3)
  * y[11] : Sin = inorganic nitrogen (kmole N/m3)
  * y[12] : Si = soluble inerts (kg COD/m3)
  * y[13] : Xc = composites (kg COD/m3)
  * y[14] : Xch = carbohydrates (kg COD/m3)
  * y[15] : Xpr = proteins (kg COD/m3)
  * y[16] : Xli = lipids (kg COD/m3)
  * y[17] : Xsu = sugar degraders (kg COD/m3)
  * y[18] : Xaa = amino acid degraders (kg COD/m3)
  * y[19] : Xfa = LCFA degraders (kg COD/m3)
  * y[20] : Xc4 = valerate and butyrate degraders (kg COD/m3)
  * y[21] : Xpro = propionate degraders (kg COD/m3)
  * y[22] : Xac = acetate degraders (kg COD/m3)
  * y[23] : Xh2 = hydrogen degraders (kg COD/m3)
  * y[24] : Xi = particulate inerts (kg COD/m3)
  * y[25] : scat+ = cations (metallic ions, strong base) (kmole/m3)
  * y[26] : san- = anions (metallic ions, strong acid) (kmole/m3)
  * y[27] : pH = pH within AD system
  * y[28] : S_H+ = protons (kmole/m3)
  * y[29] : Sva- = valerate (kg COD/m3)
  * y[30] : Sbu- = butyrate (kg COD/m3)
  * y[31] : Spro- = propionate (kg COD/m3)
  * y[32] : Sac- = acetate (kg COD/m3)
  * y[33] : Shco3- = bicarbonate (kmole C/m3)
  * y[34] : Sco2 = carbon dioxide (kmole C/m3)
  * y[35] : Snh3 = ammonia (kmole N/m3)
  * y[36] : Snh4+ = ammonium (kmole N/m3)
  * y[37] : Sgas,h2 = hydrogen concentration in gas phase (kg COD/m3)
  * y[38] : Sgas,ch4 = methane concentration in gas phase (kg COD/m3)
  * y[39] : Sgas,co2 = carbon dioxide concentration in gas phase (kmole C/m3)
  * y[40] : pgas,h2 = partial pressure of hydrogen gas (bar)
  * y[41] : pgas,ch4 = partial pressure of methane gas (bar)
  * y[42] : pgas,co2 = partial pressure of carbon dioxide gas (bar)
  * y[43] : pgas,total = total head space pressure (H2+CO2+CH4+H2O) (bar)
  * y[44] : qgas = gas flow rate normalised to atmospheric pressure (m3/d)
  *Computed with ASM interface
  * y[45] : Si = soluble inert organic material (g COD/m3)
  * y[46] : Ss = readily biodegradable substrate (g COD/m3)
  * y[47] : Xi = particulate inert organic material (g COD/m3)
  * y[48] : Xs = slowly biodegradable substrate (g COD/m3)
  * y[49] : Xd = particulate product arising from biomass decay (g COD/m3)
  * y[50] : Snh = ammonia and ammonium nitrogen (g N/m3)
  * y[51] : Sns = soluble biodegradable organic nitrogen (g N/m3)
  * y[52] : Xns = particulate biodegradable organic nitrogen (g N/m3)
  * y[53] : Salk = alkalinity (mole HCO3-/m3)
  * y[54] : TSS = total suspended solids (internal use) (mg SS/l)
  * y[55] : RT = retention time (d)
  * y[56] : ACN = acetate capacity number (kg COD/m3/d)/(kg COD/m3/d)
  */

  PetscErrorCode    ierr;
  const PetscScalar *x, *interface_params;
  PetscScalar       *y;
  PetscReal  fnaa, fnxc, fnbac, fnxi, fnsi_asm, fnsi_adm, fdegradeXbac_asm, fdegradeXI_asm;
  PetscReal  R, T_base, T_op, pK_w_base, pK_a_va_base, pK_a_bu_base, pK_a_pro_base, pK_a_ac_base, pK_a_co2_base, pK_a_IN_base;
  PetscReal  pH, pK_a_co2, pK_a_IN, alfa_va, alfa_bu, alfa_pro, alfa_ac, alfa_co2, alfa_IN, factor;
  PetscReal  Xdtemp, XStemp, XStemp2;
  PetscReal  biomass, biomass_inert, biomass_degradeN, remainCOD, inertX, noninertX, inertS, utemp[24];
  PetscInt i;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(ctx->interface_params,&interface_params);CHKERRQ(ierr);

  fnsi_asm = interface_params[0];				/*ASM's N content of SI*/
  fdegradeXbac_asm = interface_params[1];   	/*AD biomass that can be aerobically degraded  */
  fdegradeXI_asm = interface_params[2];   	/*amount of XI_adm that can be aerobically degraded*/
  fnaa = interface_params[3];					/*N content of amino acids and protein */
  fnxc = interface_params[4];					/*N content of composite material */
  fnbac = interface_params[5];				/*N content of biomass*/
  fnxi = interface_params[6];					/*N content of particulate inerts (for ASM = XI, Xd)*/
  fnsi_adm = interface_params[7];				/*ADM's N content of SI*/
  R = interface_params[8];
  T_base = interface_params[9];
  T_op = interface_params[10];
  pK_w_base = interface_params[11];
  pK_a_va_base = interface_params[12];
  pK_a_bu_base = interface_params[13];
  pK_a_pro_base = interface_params[14];
  pK_a_ac_base = interface_params[15];
  pK_a_co2_base = interface_params[16];
  pK_a_IN_base = interface_params[17];

  ierr = VecRestoreArrayRead(ctx->interface_params,&interface_params);CHKERRQ(ierr);

  ierr = VecGetArrayRead(ctx->adm1_output,&x);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->asm1_output,&y);CHKERRQ(ierr);

  for (i = 0; i < 24; i++) 	/*ADM1 values needed for interface calculations*/
    utemp[i] = x[i];

  for (i = 0; i < 26; i++) 	/*values not changed (calculated with ADM1 S-block); */
    y[i] = x[i];
    
  for (i = 26; i < 44; i++) 	/*do not output Q, T, dummy values*/
    y[i] = x[i+7];
  
  for (i = 44; i < 56; i++) 	/*adm2asm interface values (initial=0.0)*/
    y[i] = 0.0;
  
  y[54] = x[52];				/*retention time*/
  y[55] = x[53];				/*ACN indicator*/


  /*================================================================================================*/
  /* Biomass becomes XD and part of XS when transformed into ASM. Assume nitrogen portion of XS = fnxc and nitrogen portion of of XD = fnxi. If insufficient nitrogen is available from biomass, inorganic nitrogen (S_IN) can be used and extra N (e.g., from XS) becomes S_IN (which ultimately becomes Snh) */
  biomass = 1000.0*(utemp[16] + utemp[17] + utemp[18] + utemp[19] + utemp[20] + utemp[21] + utemp[22]); /* mg COD/L; Xsu Xaa Xfa Xc4 Xpro Xac Xh2 ; convert kg COD/m3 to mg COD/L using 1000 biomass = biomass_inert + biomass_degrade */
  biomass_inert = biomass*(1.0 - fdegradeXbac_asm);   /* mg COD/L; biomass_inert = Xd = fraction of AD biomass not aerobically degradable; fdegradeXbac_asm = aerobically degradable fraction of AD biomass*/
  biomass_degradeN = (biomass*fnbac - biomass_inert*fnxi);  /* mg N/L; amount of nitrogen associated with aerobically degradable biomass; COD basis:  biomass_degrade = biomass (total) - biomass_intert */
  remainCOD = 0.0;
  
  /* To avoid N errors; parameters are bounded so fnbac > fnxi and fnbac > fnxc; therefore Xd = biomas_inert and XS = biomass - biomass_inert */
  if (biomass_degradeN < 0.0) {
    /* Problem:  not enough biomass N to form Xd (inert part of AD biomass); would only happen when fnbac < fnxi*(1-fdegradeXbac_asm)  */
    printf("%s \n","WARNING: biomass_degradeN < 0, not enough biomass N to map the inert part of biomass");
    /* Take all biomass N and put into Xd */
    Xdtemp = biomass*fnbac/fnxi; /* mg COD/L; all N from biomass (=biomass*fnbac) converted to COD as Xd (=1/fnxi); max amount of Xd based on N limit */
    biomass_inert = Xdtemp;
    biomass_degradeN = 0.0; /*all N as Xd*/
  }
  else  {
    Xdtemp = biomass_inert;
  }
  
  /* Compare mg COD/L of boimass_degrade calculated using the (N approach) and (COD approach);*/
  if ((biomass_degradeN/fnxc) <= (biomass - biomass_inert)) {
    /*Problem:  if biomass_degrade calculated using N is less than concentration calculated using COD, then there is not enough N to form biomass_degrade; will not happen if fnbac>fnxi and fnbac>fnxc; biomass_degrade calculated by nitrogen balance <= biomass_degrade calculated by COD balance */
    XStemp = biomass_degradeN/fnxc;        /* since N is limiting factor then calc biomass_degrade (or XStemp) using remaining biomass N */
    remainCOD = biomass - biomass_inert - XStemp; /*mg COD/L*/
    /* now need to excess nitrogen from S_IN to combine with this excess COD to make biomass_degrade (or XStemp) */
    if ((utemp[10]*14000.0) >= remainCOD/fnxc) {  /* checking if mgN/L available from S_IN is enough N needed to map the remainCOD into biomass_degrade; convert S_IN from kmole/m3 to mg N/L with 14000 */
      XStemp = XStemp + remainCOD; /* there is enough N; amount of nitrogen needed to make remainCOD intio biomass_degrade is remobed with utemp[10] equation below */
    }
    else {
      /* 'ERROR: not enough nitrogen to map the requested XS part of biomass' */
      SETERRQ(PETSC_COMM_SELF,1, "System Failure! Not enough nitrogen to map the requested XS part of biomass.\n");
        // y[0] = 1.0;
    }
  }
  else {
    XStemp = biomass - biomass_inert; /* since COD is limiting faction then calc biomass_degrade (or XS temp) using remaining biomass COD */
  }
  
  utemp[10] = utemp[10] + biomass*fnbac/14000.0 - Xdtemp*fnxi/14000.0 - XStemp*fnxc/14000.0;  /* any remaining biomass N not going to Xd or XS goes to S_IN; convert mgN/L to kmoleN/m3 with 1/14000 */
  y[47] = (utemp[12] + utemp[13] + utemp[14] + utemp[15])*1000.0 + XStemp;     /* XS (slowly biodegradable substrate) = Xc, Xch, Xpr, Xli and XStemp; convert kg COD/m3 to mg COD/L by 1000 */
  y[48] = Xdtemp;      /* Xd = inert part of biomass */
  
  
  /*================================================================================================*/
  /* mapping of inert XI(AD) into XI(AS) and XS(AS)
    * assumption: same N content in both ASM1 and ADM1 particulate inerts (fnxi)
    * special case: if part of XI(AD) can be degraded by AS then mapped to XS(AS)
    * if N content of XS greater than XI (fnxi > fnxc), then take N from S_IN */
  inertX = (1-fdegradeXI_asm)*utemp[23]*1000.0; /*mgCOD/L of XI that canNOT be degraded by activated sludge (AS)*/
  XStemp2 = 0.0;  /*degradable portion of XI(AD) with N as fnxc*/
  noninertX = 0.0;  /*degradable portion of XI(AD) with N as fnxi */
  /* XStemp2 and noninertX are the same but kept separate until N content correction for XS(AS) derived from XI(AD) */
  
  if (fdegradeXI_asm > 0.0) {  /*If some of X_I from AD can be degraded by AS*/
    noninertX = fdegradeXI_asm*utemp[23]*1000.0;  /*mgCOD/L of X_I that AS can degrade*/
    if (fnxi < fnxc)  {     /* if N in XI(AD) is not enough for XS(AS) */
      XStemp2 = noninertX*fnxi/fnxc;  /*mg COD XS/L; max XS possible based on N in XI*/
      noninertX = noninertX - noninertX*fnxi/fnxc; /*noninertX = remaining COD (COD that didn't have enough N to become XS)*/
      if ((utemp[10]*14000.0) < (noninertX*fnxc))  {  /* if S_IN < N needed to convert noninertX to XS */
        XStemp2 = XStemp2 + (utemp[10]*14000.0)/fnxc;  /*use all S_IN nitrogen to form max amount of XS from noninertX based on N*/
        noninertX = noninertX - (utemp[10]*14000.0)/fnxc;
        utemp[10] = 0.0;  /*used all S_IN*/
        /* Nitrogen shortage when converting biodegradable XI; map remaining noninertX to XI(AS) */
        inertX = inertX + noninertX;
      }
      else  {   /* there is enough N in S_IN to convert all noninertX into XS */
        XStemp2 = XStemp2 + noninertX;
        utemp[10] = utemp[10] - noninertX*fnxc/14000.0;  /*N used for XS mapping removed from S_IN*/
        noninertX = 0.0;
      }
    }
    else  {   /* N in XI(AD) enough for mapping */
      XStemp2 = XStemp2 + noninertX;
      utemp[10] = utemp[10] + noninertX*(fnxi - fnxc)/14000.0;    /* put remaining N as S_IN */
      noninertX = 0.0;
    }
  }
  
  y[46] = inertX;          /* Xi = particulate inert organic matter */
  y[47] = y[47] + XStemp2;  /* XS  */
  
  /*================================================================================================*/
  /* Mapping of SI(AD) to SI(AS)
    * It is assumed that this mapping will be 100% on COD basis
    * if N content of SI(AS) > SI(AD), then take N from S_IN. */
  
  inertS = 0.0;
  if (fnsi_adm < fnsi_asm) {   /* if N in SI(AD) not enough */
    inertS = utemp[11]*fnsi_adm/fnsi_asm; /*max possible SI(AD) kg COD/m3 based on N limitation*/
    utemp[11] = utemp[11] - utemp[11]*fnsi_adm/fnsi_asm;
    if ((utemp[10]*14.0) < (utemp[11]*fnsi_asm))  {  /* if N in S_IN not enough; kg N/m3 from S_IN < kg N/m3 from SI(AD) */
      inertS = inertS + utemp[10]*14/fnsi_asm; /*kg COD/m3*/
      utemp[11] = utemp[11] - utemp[10]*14.0/fnsi_asm; /*SI(AD) minus amount of COD converted to SI(AS) based on available N from S_IN*/
      utemp[10] = 0.0; /*All N from S_IN used*/
      /* ERROR: Nitrogen shortage when converting SI */
      SETERRQ(PETSC_COMM_WORLD,1,"System failure: nowhere to put SI\n");
      // y[0] = 1.0;
        /*  bound parameter so that fnsi_adm >= fnsi_asm  */
    }
  else  {  /* N in S_IN enough for mapping SI(AD) to SI(AS)*/
    inertS = inertS + utemp[11];
    utemp[10] = utemp[10] - utemp[11]*fnsi_asm/14.0;
    utemp[11] = 0.0;
    }
  }
  else  {    /* N in SI(AD) enough for mapping */
    inertS = inertS + utemp[11]; /*all SI(AD) becomes SI(AS)*/
    utemp[10] = utemp[10] + utemp[11]*(fnsi_adm - fnsi_asm)/14.0;  /* put remaining N as S_IN; kmole N/m3 */
    utemp[11] = 0.0;
  }
  
  y[44] = inertS*1000.0;		/* Si; converted kg COD/m3 to mg COD/L with 1000*/
  
  /*================================================================================================*/
  /* sh2 and sch4 assumed to be stripped upon reentry to ASM side */
  /* Soluble Substrate, nitrogen, and charge balance */
  
  y[45] = (utemp[0] + utemp[1] + utemp[2] + utemp[3] + utemp[4] + utemp[5] + utemp[6])*1000.0;	/* Ss = Ssu, Saa, Sfa, Sva, Sbu, Spro, Sac */
  
  y[49] = utemp[10]*14000.0;		/* Snh = NH3 and NH4+ = S_IN including adjustments above; convert kmoleN/m3 to mgN/l */
  
  /* Sns is the nitrogen part of Ss in ASM1.
    * Nirogen is from the amino acids. In ASM1, the N content of Si is not included in Sns. */
  y[50] = fnaa*1000.0*utemp[1];
  
  /* Xns represents the nitrogen part of XS in ASM1.
    * Nitrogen is from biomass (XStemp), degradable XI (XStemp2), composites, and proteins (assume no nitrogen in carbohydrates and lipids). The N content of XI(AS) is not included in Xns in ASM1. */
  y[51] = fnxc*(XStemp + XStemp2 + 1000.0*utemp[12]) + fnaa*1000.0*utemp[14]; /*Xns = nitrogen from AD biomass + degradable XI(AD) + composites + proteins */
  
  y[53] = 0.75*(y[46] + y[47] + y[48]); /*TSS (mg SS/l) = XI + XS + Xd*/
  
  /* S_alk (molHCO3/m3), charge balance; VFA converted from kgCOD to kmol with alfa; subtract positive charge (ammonium); add negative charge (HCO3-, VFA);  convert ALK from Kmol/m3 into mol/m3; */
  /* van't Hoff eqn for temperature dependence */
  factor = (1.0/T_base - 1.0/T_op)/(100.0*R); /*units = mol/J; 100 to convert R (bar/M/K to J/mol/K)*/
  pK_a_co2 = pK_a_co2_base - log10(exp(7600.0*factor));
  pK_a_IN = pK_a_IN_base - log10(exp(51800.0*factor));
  /* Degree of Ionization */
  pH = x[33];
  alfa_va = 1.0/208.0*(1.0 + pow(10, pK_a_va_base - pH)); /*includes conversion of kgCOD to kmole (208 kgCOD/kmole)*/
  alfa_bu = 1.0/160.0*(1.0 + pow(10, pK_a_bu_base - pH));
  alfa_pro = 1.0/112.0*(1.0 + pow(10, pK_a_pro_base - pH));
  alfa_ac = 1.0/64.0*(1.0 + pow(10, pK_a_ac_base - pH));
  alfa_co2 = 1.0/(1.0 + pow(10, pK_a_co2 - pH)); /*HCO3- = alfa_co2*S_IN; simplified; assume HCO3- >> CO3-2*/
  alfa_IN = 1.0/(1.0 + pow(10, pK_a_IN - pH)); /*NH3 = alfa_IN*S_IN*/
  /* Alkalinity Calculation (charge balance; alkalinity is how much negative charge remains after neutralizing all the positive charges) */
  y[52] = (x[3]*alfa_va + x[4]*alfa_bu + x[5]*alfa_pro + x[6]*alfa_ac + x[9]*alfa_co2 - x[10]*(1-alfa_IN) - (-y[49]/14000.0*(1-alfa_IN)))*1000.0;

  ierr = VecRestoreArrayRead(ctx->adm1_output,&x);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->asm1_output,&y);CHKERRQ(ierr);

  PetscFunctionReturn(0);

}


PetscErrorCode ProcessIndicators(AppCtx *ctx)
{
  /* Converts ASM1 output to indicators */

  /*
  * Output vector:
 * y[0] : Ssu = monosaccharides (mg COD/L)
 * y[1] : Saa = amino acids (mg COD/L)
 * y[2] : Sfa = long chain fatty acids (LCFA) (mg COD/L)
 * y[3] : Sva = total valerate (mg COD/L)
 * y[4] : Sbu = total butyrate (mg COD/L)
 * y[5] : Spro = total propionate (mg COD/L)
 * y[6] : Sac = total acetate (mg COD/L)
 * y[7] : Sh2 = hydrogen gas (mg COD/L)
 * y[8] : Sch4 = methane gas (mg COD/L)
 * y[9] : Sic = inorganic carbon (mg C/L)
 * y[10] : Sin = inorganic nitrogen (mg N/L)
 * y[11] : Si = soluble inerts (mg COD/L)
 * y[12] : Xc = composites (mg COD/L)
 * y[13] : Xch = carbohydrates (mg COD/L)
 * y[14] : Xpr = proteins (mg COD/L)
 * y[15] : Xli = lipids (mg COD/L)
 * y[16] : Xsu = sugar degraders (mg COD/L)
 * y[17] : Xaa = amino acid degraders (mg COD/L)
 * y[18] : Xfa = LCFA degraders (mg COD/L)
 * y[19] : Xc4 = valerate and butyrate degraders (mg COD/L)
 * y[20] : Xpro = propionate degraders (mg COD/L)
 * y[21] : Xac = acetate degraders (mg COD/L)
 * y[22] : Xh2 = hydrogen degraders (mg COD/L)
 * y[23] : Xi = particulate inerts (mg COD/L)
 * y[24] : scat+ = cations (metallic ions, strong base) (mmol/L)
 * y[25] : san- = anions (metallic ions, strong acid) (mmol/L)
 * y[26] : pH = pH within AD system   
 * y[27] : S_H+ = protons (mol/L)
 * y[28] : Sva- = valerate (mg COD/L)
 * y[29] : Sbu- = butyrate (mg COD/L)
 * y[30] : Spro- = propionate (mg COD/L)
 * y[31] : Sac- = acetate (mg COD/L)
 * y[32] : Shco3- = bicarbonate (mg C/L)
 * y[33] : Sco2 = carbon dioxide (mg C/L)
 * y[34] : Snh3 = ammonia (mg N/L)
 * y[35] : Snh4+ = ammonium (mg N/L)
 * y[36] : Sgas,h2 = hydrogen concentration in gas phase (mg COD/L)
 * y[37] : Sgas,ch4 = methane concentration in gas phase (mg COD/L)
 * y[38] : Sgas,co2 = carbon dioxide concentration in gas phase (mg C/L)
 * y[39] : pgas,h2 = partial pressure of hydrogen gas (atm)
 * y[40] : pgas,ch4 = partial pressure of methane gas (atm)
 * y[41] : pgas,co2 = partial pressure of carbon dioxide gas (atm)
 * y[42] : pgas,total = total head space pressure (H2+CO2+CH4+H2O) (atm)
 * y[43] : qgas = gas flow rate normalised to atmospheric pressure (m3/d)
 * y[44] : Si = soluble inert organic material (mg COD/L)
 * y[45] : Ss = readily biodegradable substrate (mg COD/L)
 * y[46] : Xi = particulate inert organic material (mg COD/L)
 * y[47] : Xs = slowly biodegradable substrate (mg COD/L)
 * y[48] : Xd = particulate product arising from biomass decay (mg COD/L)
 * y[49] : Snh = ammonia and ammonium nitrogen (mg N/L)
 * y[50] : Sns = soluble biodegradable organic nitrogen (mg N/L)
 * y[51] : Xns = particulate biodegradable organic nitrogen (mg N/L)
 * y[52] : Salk = alkalinity (mg C/L) calculated with adm2asm interface
 * y[53] : TSS = total suspended solids (internal use) (mg TSS/L)
Indicators
 * y[54] : totalVFA = total VFA concentration (mg COD/L)
 * y[55] : mass_Sac = mass acetate (mg acetate/L)
 * y[56] : PAratio = propionate to acetate ration (kg acetate/ kg acetate equivalent of propionate)
 * y[57] : Alk = alkalinity (mg/L CaCO3) calculated based on ADM1 outputs
 * y[58] : NH3 = ammonia (mg N/L)
 * y[59] : NH4 = ammonium (mg N/L)
 * y[60] : LCFA = long chain fatty acids (mg COD/L)
 * y[61] : percentch4 = biogas methane content (percent by volume; percent not decimal)
 * y[62] : energy = energy content of methane gas (MJ/d); no longer used as indicator
 * y[63] : efficiency = COD removal (%)
 * y[64] : VFA_Alk = VFA to alkalinity ratio (mg acetate eq./L )/(mg CaCO3 eq./L)
 * y[65] : ACN = acetate capacity number (kg COD/m3/d)/(kg COD/m3/d)
 * y[66] : RT = retention time (d)
  */

  PetscErrorCode    ierr;
  const PetscScalar *x, *params, *influent;
  PetscScalar       *y;
  PetscInt i;
  PetscReal  patm, R, T_base, T_op, pK_w_base, pK_a_va_base, pK_a_bu_base, pK_a_pro_base, pK_a_ac_base, pK_a_co2_base, pK_a_IN_base;
	PetscReal  factor, pK_w, pK_a_co2, pK_a_IN, pH, alfa_va, alfa_bu, alfa_pro, alfa_ac, alfa_co2, alfa_IN;
	PetscReal WWTP_energy, qheat, Eprod, Eheat, Eelec, CODout, CODin;
  PetscReal xtemp[89];


  PetscFunctionBegin;
  ierr = VecGetArrayRead(ctx->params,&params);CHKERRQ(ierr);
  ierr = VecGetArrayRead(ctx->asm1_output,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(ctx->influent,&influent);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->indicator,&y);CHKERRQ(ierr);

  /*Concatenate the two vectors asm1_output and influent */
  for (i = 0; i < 56; i++) 	/* asm1_output */
    xtemp[i] = x[i];

  for (i = 56; i < 89; i++) 	/* influent values */
    xtemp[i] = influent[i-56];


  /*Input values not changed; u[0] to u[53]*/
	for (i = 0; i < 39; i++){ /* kgCOD/m3*1000=mgCOD/L; kmol/m3*1000=mmol/L for ions */
		y[i] = xtemp[i]*1000.0; 
	}
	/* except the following */ 
  /*Carbon*/
  y[9] = xtemp[9]*12000.0;		/*Sic; kmolC/m3*12000=mgC/L*/
  y[32] = xtemp[32]*12000.0;		/*Shco3; kmolC/m3*12000=mgC/L*/
  y[33] = xtemp[33]*12000.0;		/*Sco2; kmolC/m3*12000=mgC/L*/
  y[38] = xtemp[38]*12000.0;		/*Sgas,co2; kmolC/m3*12000=mgC/L*/
  /* Nitrogen*/
  y[10] = xtemp[10]*14000.0;		/*Sin; kmolN/m3*14000=mgN/L*/
  y[34] = xtemp[34]*14000.0;		/*Snh3; kmolN/m3*14000=mgN/L*/
  y[35] = xtemp[35]*14000.0;		/*Snh4+; kmolN/m3*14000=mgN/L*/
  y[26]=xtemp[26];				/* pH */
  y[27]=xtemp[27];				/* protons; mol/L */
	
	for (i = 39; i < 43; i++){ /* gas pressure; bar/1.01325=atm */
		y[i] = xtemp[i]/1.01325; 
	}
	for (i = 43; i < 54; i++){ /* no unit change, except y[52] */
		y[i] = xtemp[i]; 
	}	
  y[66] = xtemp[54];			/*retention time, d*/
	
		
	/* Indicator Calculations */
	/* DIGESTERPAR */
  patm = params[93]; /*bar*/
  R = params[77];
  T_base = params[78];
  T_op = params[79];
  pK_w_base = params[80];
  pK_a_va_base = params[81];
  pK_a_bu_base = params[82];
  pK_a_pro_base = params[83];
  pK_a_ac_base = params[84];
  pK_a_co2_base = params[85];
  pK_a_IN_base = params[86];
	
	/* van't Hoff eqn for temperature dependence */
  factor = (1.0/T_base - 1.0/T_op)/(100.0*R); /*units = mol/J; 100 to convert R (bar/M/K to J/mol/K)*/
  pK_w = pK_w_base - log10(exp(55700.0*factor));  /*enthalphy of reaction= 55700 J/mol*/
  pK_a_co2 = pK_a_co2_base - log10(exp(7600.0*factor));
  pK_a_IN = pK_a_IN_base - log10(exp(51800.0*factor));
    
	/* Degree of Ionization */
  pH = xtemp[26];
  alfa_va = 1.0/(1.0 + pow(10, pK_a_va_base - pH)); 
  alfa_bu = 1.0/(1.0 + pow(10, pK_a_bu_base - pH));
  alfa_pro = 1.0/(1.0 + pow(10, pK_a_pro_base - pH));
  alfa_ac = 1.0/(1.0 + pow(10, pK_a_ac_base - pH));
  alfa_co2 = 1.0/(1.0 + pow(10, pK_a_co2 - pH)); /*simplified; assume HCO3- >> CO3-2*/
  alfa_IN = 1.0/(1.0 + pow(10, pK_a_IN - pH)); /*NH3 = alfa_IN*INtotal*/


  /* VFA
	* VFA_C2toC5 = 3.7 (g COD/L) or (kg COD/m3)         [Ferrer et al. (2010) Bioresour. Technol.]
	* acetate = 0.6 (g acetate/L) or (kg ac/m3)         [Ferrer et al. (2010) Bioresour. Technol.]
	* PAratio  = TBD (kg propionate as acetate equivalent/kg acetate)    [TBD]
	*/
  y[54] = (xtemp[3]+xtemp[4]+xtemp[5])*1000.0; 			/*VFA_C2toC5 = Sva+Sbu+Spro; (mg COD/L) */
	y[55] = xtemp[6]/64.0*60.0*1000.0; 			/*mass_Sac (kg acetate/m3) = (kgCOD/m3)*(60 kg/kmol)/(64 kgCOD/kmol)*(1000 mg/kg*m3/L) */
	y[56] = xtemp[5]/(xtemp[6]);  				/*PAratio = Spro/Sac; (kg COD as propionate/kg COD as acetate) */
	
	
  /* Alkalinity
	* Alkalinity = 2-4 (g/L CaCO3) minimum           [GDLF]
	* Total Alkalinity = Bicarbonate Alk (mg/L CaCO3) + 0.71*S_VFA (total VFA concentration as acetic acid)	[GDLF p584]
	* Nitrogen Alkalinity:  1 mg N = 3.6 mg CaCO3    [Paramewwaran& Rittman (2012) Bioresour. Technol.]
	* S_alk (molHCO3/m3), charge balance; assume HCO3- >> CO32-; VFA converted from kgCOD to Kmole
	* Alk (g/L CaCO3); convert kmol/m3 to mg/L CaCO3(100 g/mol CaCO3 * 1000 mg/kg*m3/L )
	*/
 	y[52] = xtemp[52]*12.0;	/*Salk; moleC/m3*12=mgC/L */
	y[57] = xtemp[52]*100.0;	/*Salk; moleC/m3*100=mgCaCO3/L */
		/* [Nopens et al. Water Res (2009)] Salk for ASM does not include pH or Snh (p.1919) */
	
	
	/* Nitrogen
	* Snh3 = 150 mg NH3-N/L  [Paramewwaran & Rittman (2012) Bioresour. Technol.]
	* Snh4+ = 5 g NH4-N/L    [Paramewwaran & Rittman (2012) Bioresour. Technol.]
	*/
	y[58] = xtemp[34]*1000.0*14.0; 				/*NH3 (mg N/L); converted from kmole N/m3*/
	y[59] = xtemp[35]*1000.0*14.0; 				/*NH4 (mg N/L); converted from kmole N/m3*/
	
	
	/* LCFA effluent concentration
	* Sfa = 180-220 g COD-LCFA/kg TS reactor concentration[Neves et al. (2009) Wat Res]
	* TS = TSS+TDS; conservative estimate of TS~TSS
	* Using TSS in denominator will overestimate COD value (conservative); 
	* Use higher LCFA range (220 gCOD/kg TS) to offset overestimation
	*/
	y[60] = y[2]; 			/*LCFA (mg COD/L)*/

	
	/*Biogas
	* percent methane = 55%      [Ferrer et al. (2010) Bioresour. Technol.]
	* Methane properties:  1kg CH4 = 50.014 MJ; 64 g COD/mole CH4; 16 g CH4/mole CH4
	* using ideal gas law:  Vch4/pgas_ch4 = V_gas/pgas_total --> Vch4/V_gas = pgas_ch4/pgas_total
	* percentch4 = pgas_ch4/pgas_total; 
	* energy = Sgas,ch4/(64kgCOD/kmole)*(16kg/kmole)*(50.014MJ/kg)*qgas
	* flow = qgas * percentch4
	*/
	y[61] = xtemp[40]/(xtemp[42])*100.0; 				/*percentch4 (percent by volume); percent not decimal*/
	/*y[63] = xtemp[43]*(y[61]/100.0); */			/*flowch4 (m3 methane/d; normalized to patm) (not used)*/
	
    
    	/*Energy Offset
	*(kJ/d) based on 6.4 MGD WWTP with activated sludge (producing about 170 m3/d of thickened waste activated and primary sludges, which requires a digester volume about 3400m3) and typical energy demand of 1100 MJ/1000m3 wastewater & heat energy for digester (energy calc spreadsheet for more detail)
	*energy from biogas goes first to digester heating (heating efficiency 0.75), remaining to electricity (conversion efficiency = 0.35)
	*y[62] = xtemp[37]/64.0*16.0*50.014*(xtemp[43]);	//energy (MJ/d) theoretical maximum from methane concentration
	*/
	WWTP_energy = 42000000.0; /*kJ/d for WWTP/activated sludge + digester heating (40 GJ)*/
	qheat = 770.0; /*m3/d for average biogas needed to heating digester*/
	if (xtemp[43] < qheat) {/*xtemp[43] is qgas in m3/d*/
		Eheat = xtemp[43]*(0.75*23000.0);	/*kJ/d; heating value of digester gas=23000 kJ/m3, heating efficiency=0.75*/
		Eprod = Eheat;		/*kJ/d*/
	}
	else {
		Eheat = qheat*(0.75*23000.0);	/*kJ/d; heating value of digester gas=23000 kJ/m3, heating efficiency=0.75*/
		Eelec = (xtemp[43]-qheat)*(0.35*23000.0); /*kJ/d; electrical conversion efficiency=0.35*/
		Eprod = Eheat + Eelec;	/*kJ/d*/
	}
	y[62] = Eprod/WWTP_energy*100.0; /*energy (% offset)*/
	
	/*Efficiency (COD conversion)
    *The amount of COD removal; 1-CODout/CODin; 
     *CODout = all soluble and particulate components #0-8 and 11-23
     *CODin = all soluble and particulate components #
     */
  CODout = xtemp[0]+xtemp[1]+xtemp[2]+xtemp[3]+xtemp[4]+xtemp[5]+xtemp[6]+xtemp[7]+xtemp[8]+xtemp[11]+xtemp[12]+xtemp[13]+xtemp[14]+xtemp[15]+xtemp[16]+xtemp[17]+xtemp[18]+xtemp[19]+xtemp[20]+xtemp[21]+xtemp[22]+xtemp[23];
  CODin = xtemp[56]+xtemp[57]+xtemp[58]+xtemp[59]+xtemp[60]+xtemp[61]+xtemp[62]+xtemp[63]+xtemp[64]+xtemp[67]+xtemp[68]+xtemp[69]+xtemp[70]+xtemp[71]+xtemp[72]+xtemp[73]+xtemp[74]+xtemp[75]+xtemp[76]+xtemp[77]+xtemp[78]+xtemp[79];
  y[63] = (1.0-(CODout/CODin))*100.0; /*% on COD basis*/
	if (y[63]<0) {/*if value is approaching zero [possible for it to be a small negative value] then force to  zero*/
    if (y[63]>-0.01) {
        y[63]=0;
      }
  }
	
	
	/* VFA to Alkalinity ratio, (mg acetate eq./L )/(mg CaCO3 eq./L) 
	* Total VFA = VFA = = Sva+Sbu+Spro+Sac (mg COD/L)
	* Acetate equivalents = (mgCOD/L) * (60 mgHAc/mmolHAc) / (64 mgCOD/mmol HAc)
	* VFA/Alk = 0.1-0.4 favorable operating conditions 		[Schoen et al. (2009) Bioresour. Technol.]
	* >0.4 indicate upset and need for corrective action	[Schoen et al. (2009) Bioresour. Technol.]
	* >0.8 indicate process can fail						[Schoen et al. (2009) Bioresour. Technol.]
	*/
	y[64] = ((y[54]+xtemp[6]*1000.0)*60.0/64.0)/y[57];
	
	
	/* ACN indicator (calculated in ADM1)  
	* ACN = maximum acetate utilization rate/ acetate production rate (kg COD/m3/d)/(kg COD/m3/d)
	* ACN < 1 (at steady-state) indicate impending failure [Schoen et al. (2009) Bioresour. Technol.]
	*/	
	y[65] = xtemp[55];

  ierr = VecRestoreArrayRead(ctx->params,&params);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(ctx->asm1_output,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(ctx->influent,&influent);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->indicator,&y);CHKERRQ(ierr);

  PetscFunctionReturn(0);

}


PetscErrorCode PostProcess(TS ts)
{
  PetscErrorCode ierr;
  PetscViewer    viewer;
  AppCtx         *ctx;
  Vec            adm1_sol;
  PetscInt       stepi;
  char           filename[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(ts,&ctx);
  ierr = TSGetSolution(ts,&adm1_sol);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&stepi);CHKERRQ(ierr);
  ierr = ProcessADM1(adm1_sol,ctx);CHKERRQ(ierr);
  ierr = PetscSNPrintf(filename,sizeof filename,"%s-%03D.out","adm1_output",stepi);CHKERRQ(ierr);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename,&viewer);CHKERRQ(ierr);
  ierr = VecView(ctx->adm1_output,viewer);CHKERRQ(ierr);
  ierr = ProcessADM1toASM1(ctx);CHKERRQ(ierr);
  ierr = ProcessIndicators(ctx);CHKERRQ(ierr);
  ierr = PetscSNPrintf(filename,sizeof filename,"%s-%03D.out","indicator",stepi);CHKERRQ(ierr);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename,&viewer);CHKERRQ(ierr);
  ierr = VecView(ctx->indicator,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);

}

/*
     Defines the ODE passed to the ODE solver
*/
PetscErrorCode IFunctionPassive(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,AppCtx *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *f;
  const PetscScalar *x,*xdot,*params,*influent; 
  PetscReal f_sI_xc, f_xI_xc, f_ch_xc, f_pr_xc, f_li_xc, N_xc, N_I, N_aa, C_xc, C_sI, C_ch, C_pr, C_li, C_xI, C_su, C_aa, f_fa_li, C_fa, f_h2_su, f_bu_su, f_pro_su, f_ac_su, N_bac, C_bu, C_pro, C_ac, C_bac, Y_su, f_h2_aa, f_va_aa, f_bu_aa, f_pro_aa, f_ac_aa, C_va, Y_aa, Y_fa, Y_c4, Y_pro, C_ch4, Y_ac, Y_h2;
  PetscReal k_dis, k_hyd_ch, k_hyd_pr, k_hyd_li, K_S_IN, k_m_su, K_S_su, pH_UL_aa, pH_LL_aa;
  PetscReal k_m_aa, K_S_aa, k_m_fa, K_S_fa, K_Ih2_fa, k_m_c4, K_S_c4, K_Ih2_c4, k_m_pro, K_S_pro;
  PetscReal K_Ih2_pro, k_m_ac, K_S_ac, K_I_nh3, pH_UL_ac, pH_LL_ac, k_m_h2, K_S_h2, pH_UL_h2, pH_LL_h2;
  PetscReal k_dec_Xsu, k_dec_Xaa, k_dec_Xfa, k_dec_Xc4, k_dec_Xpro, k_dec_Xac, k_dec_Xh2;
  PetscReal R, T_base, T_op;
  PetscReal K_H_h2_base, factor; 
  PetscReal P_atm, p_gas_h2o, P_gas, k_P, kLa, K_H_co2, K_H_ch4, K_H_h2;
  PetscReal V_liq, V_gas, t_resx, V_frac;
  PetscReal eps, pH_op, S_H_ion;
  PetscReal proc1, proc2, proc3, proc4, proc5, proc6, proc7, proc8, proc9, proc10, proc11, proc12, proc13, proc14, proc15, proc16, proc17, proc18, proc19, procT8, procT9, procT10;
  PetscReal I_pH_aa, I_pH_ac, I_pH_h2, I_IN_lim, I_h2_fa, I_h2_c4, I_h2_pro, I_nh3;
  PetscReal reac1, reac2, reac3, reac4, reac5, reac6, reac7, reac8, reac9, reac10, reac11, reac12, reac13, reac14, reac15, reac16, reac17, reac18, reac19, reac20, reac21, reac22, reac23, reac24, stoich1, stoich2, stoich3, stoich4, stoich5, stoich6, stoich7, stoich8, stoich9, stoich10, stoich11, stoich12, stoich13;
  PetscReal inhib[6]; /*declare arrays */
  PetscReal p_gas_h2, p_gas_ch4, p_gas_co2, q_gas;
  PetscReal pHLim_aa, pHLim_ac, pHLim_h2, n_aa, n_ac, n_h2;
  PetscReal proc11_max;
  PetscReal K_w, pK_w_base, K_a_va, pK_a_va_base, K_a_bu, pK_a_bu_base, K_a_pro, pK_a_pro_base, K_a_ac, pK_a_ac_base, K_a_co2, pK_a_co2_base, K_a_IN, pK_a_IN_base;

  eps = 0.0;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(ctx->params,&params);CHKERRQ(ierr);
  f_sI_xc = params[0];
  f_xI_xc = params[1];
  f_ch_xc = params[2];
  f_pr_xc = params[3];
  f_li_xc = params[4];
  N_xc = params[5];
  N_I = params[6];
  N_aa = params[7];
  C_xc = params[8];
  C_sI = params[9];
  C_ch = params[10];
  C_pr = params[11];
  C_li = params[12];
  C_xI = params[13];
  C_su = params[14];
  C_aa = params[15];
  f_fa_li = params[16];
  C_fa = params[17];
  f_h2_su = params[18];
  f_bu_su = params[19];
  f_pro_su = params[20];
  f_ac_su = params[21];
  N_bac = params[22];
  C_bu = params[23];
  C_pro = params[24];
  C_ac = params[25];
  C_bac = params[26];
  Y_su = params[27];
  f_h2_aa = params[28];
  f_va_aa = params[29];
  f_bu_aa = params[30];
  f_pro_aa = params[31];
  f_ac_aa = params[32];
  C_va = params[33];
  Y_aa = params[34];
  Y_fa = params[35];
  Y_c4 = params[36];
  Y_pro = params[37];
  C_ch4 = params[38];
  Y_ac = params[39];
  Y_h2 = params[40];
  k_dis = params[41];
  k_hyd_ch = params[42];
  k_hyd_pr = params[43];
  k_hyd_li = params[44];
  K_S_IN = params[45];
  k_m_su = params[46];
  K_S_su = params[47];
  pH_UL_aa = params[48];
  pH_LL_aa = params[49];
  k_m_aa = params[50];
  K_S_aa = params[51];
  k_m_fa = params[52];
  K_S_fa = params[53];
  K_Ih2_fa = params[54];
  k_m_c4 = params[55];
  K_S_c4 = params[56];
  K_Ih2_c4 = params[57];
  k_m_pro = params[58];
  K_S_pro = params[59];
  K_Ih2_pro = params[60];
  k_m_ac = params[61];
  K_S_ac = params[62];
  K_I_nh3 = params[63];
  pH_UL_ac = params[64];
  pH_LL_ac = params[65];
  k_m_h2 = params[66];
  K_S_h2 = params[67];
  pH_UL_h2 = params[68];
  pH_LL_h2 = params[69];
  k_dec_Xsu = params[70];
  k_dec_Xaa = params[71];
  k_dec_Xfa = params[72];
  k_dec_Xc4 = params[73];
  k_dec_Xpro = params[74];
  k_dec_Xac = params[75];
  k_dec_Xh2 = params[76];
  R = params[77];
  T_base = params[78];
  T_op = params[79];
  pK_w_base = params[80];
  pK_a_va_base = params[81];
  pK_a_bu_base = params[82];
  pK_a_pro_base = params[83];
  pK_a_ac_base = params[84];
  pK_a_co2_base = params[85];
  pK_a_IN_base = params[86];

  P_atm = params[93];
  kLa = params[94];
  K_H_h2_base = params[98];
  k_P = params[99];

  ierr = VecRestoreArrayRead(ctx->params,&params);CHKERRQ(ierr);

  V_liq = ctx->V[0];
  V_gas = ctx->V[1];
  t_resx = ctx->t_resx;

  //ierr = PetscPrintf(MPI_COMM_SELF,"SRT is: %f\n", t_resx);CHKERRQ(ierr);


  /* Set the residual */
  ierr = VecGetArrayRead(ctx->influent,&influent);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  factor = (1.0/T_base - 1.0/T_op)/(100.0*R);
  K_a_va = pow(10,-pK_a_va_base);
  K_a_bu = pow(10,-pK_a_bu_base);
  K_a_pro = pow(10,-pK_a_pro_base);
  K_w = pow(10,-pK_w_base)*exp(55700.0*factor); /* T adjustment for K_w */
  K_a_ac = pow(10,-pK_a_ac_base)*exp(-4600.0*factor);  /* T adjustment */
  K_a_co2 = pow(10,-pK_a_co2_base)*exp(7600.0*factor); /* T adjustment for K_a_co2 */
  K_a_IN = pow(10,-pK_a_IN_base)*exp(51800.0*factor);  /* T adjustment for K_a_IN */


  K_H_h2 = 1.0/pow(10,(-187.04/T_op+5.473))*55.6/1.01325;     /* atm*L/mol; modified van't Hoff-Arrhenius relationship; MetCalf & Eddy, p68; conversions 55.6 mole H2O/L; 1 atm = 1.01325 */
  K_H_ch4 = 1.0/pow(10,(-675.74/T_op+6.880))*55.6/1.01325; /* atm*L/mol; modified van't Hoff-Arrhenius relationship; MetCalf & Eddy, p68 */
  K_H_co2 = 1.0/pow(10,(-1012.40/T_op+6.606))*55.6/1.01325; /* atm*L/mol; modified van't Hoff-Arrhenius relationship; MetCalf & Eddy, p68 */

  p_gas_h2o = pow(10, (5.20389-1733.926/(T_op-39.485)));  /* Antoine equation for water pressure for temp range 31C-60C */ 

  p_gas_h2 = x[32]*R*T_op/16.0;
  p_gas_ch4 = x[33]*R*T_op/64.0;
  p_gas_co2 = x[34]*R*T_op;
  P_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o;

  S_H_ion = x[42];
  pH_op = -log10(x[42]);   /* pH */


  /* Inhibition */
  /* Hill function on SH+ used within BSM2, ADM1 Workshop, Copenhagen 2005 */
  /*Hill function is replaced by other inhibition terms, assign value to the parameters in Hill function to avoid error*/
//  pHLim_aa = pow(10,(-(pH_UL_aa + pH_LL_aa)/2.0));
//  pHLim_ac = pow(10,(-(pH_UL_ac + pH_LL_ac)/2.0));
//  pHLim_h2 = pow(10,(-(pH_UL_h2 + pH_LL_h2)/2.0));
//  n_aa=3.0/(pH_UL_aa-pH_LL_aa);
//  n_ac=3.0/(pH_UL_ac-pH_LL_ac);
//  n_h2=3.0/(pH_UL_h2-pH_LL_h2);
//  I_pH_aa = pow(pHLim_aa,n_aa)/(pow(S_H_ion,n_aa)+pow(pHLim_aa,n_aa));
//  I_pH_ac = pow(pHLim_ac,n_ac)/(pow(S_H_ion,n_ac)+pow(pHLim_ac,n_ac));
//  I_pH_h2 = pow(pHLim_h2,n_h2)/(pow(S_H_ion,n_h2)+pow(pHLim_h2,n_h2));
  pHLim_aa = 1;
  pHLim_ac = 1;
  pHLim_h2 = 1;
  n_aa=1;
  n_ac=1;
  n_h2=1;
  I_pH_aa = 1;
  I_pH_ac = 1;
  
  /*New inhibition terms used*/
  I_pH_h2 = (pH_UL_h2) / ( pH_UL_h2 + x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))*60/(64*1000) ); /*pH_UL_h2 repurposed to represent inhibition constant K_I_ac_H2, acetic acid inhibition on H2 methanogenesis*/

  I_IN_lim = 1.0/(1.0+K_S_IN/x[10]);
  I_h2_fa = 1.0/(1.0+x[7]/K_Ih2_fa);//
  I_h2_c4 = 1.0/(1.0+x[7]/K_Ih2_c4);//
  I_h2_pro = 1.0/(1.0+x[7]/K_Ih2_pro);//
  I_nh3 = 1.0/(1.0+x[31]/K_I_nh3);//

  inhib[0] = I_IN_lim;
//    I_pH_aa*I_IN_lim;
  inhib[1] = inhib[0]*I_h2_fa;
  inhib[2] = inhib[0]*I_h2_c4;
  inhib[3] = inhib[0]*I_h2_pro;
  inhib[4] = I_IN_lim*I_nh3;
//    I_pH_ac*I_IN_lim*I_nh3;
  inhib[5] = I_pH_h2*I_IN_lim;

  /* Process Rates*/  
  proc1 = k_dis*x[12]*exp(3737*(T_op-T_base)/T_op/T_base);
  proc2 = k_hyd_ch*x[13]*exp(3737*(T_op-T_base)/T_op/T_base);
  proc3 = k_hyd_pr*x[14]*exp(3737*(T_op-T_base)/T_op/T_base);
  proc4 = k_hyd_li*x[15]*exp(3737*(T_op-T_base)/T_op/T_base);
    
  proc5 = k_m_su*x[0]/(K_S_su+x[0]*(1 + (x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))) / pH_UL_aa))*x[16]*inhib[0]*exp(3101*(T_op-T_base)/T_op/T_base); /*pH_UL_aa repurposed to represent inhibition constant K_I_ac_aa, acetic acid inhibition on acetogenesis, to replace I_pH*/
  proc6 = k_m_aa*x[1]/(K_S_aa+x[1]*(1 + (x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))) / pH_UL_aa))*x[17]*inhib[0]*exp(3101*(T_op-T_base)/T_op/T_base);
  proc7 = k_m_fa*x[2]*2370*x[18]/(K_S_fa*2370*x[18]+x[2]*(1 + (x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))) / pH_UL_aa)*2370*x[18]+x[2]*x[2]*x[2])*x[18]*inhib[1]*exp(3101*(T_op-T_base)/T_op/T_base); /*LCFA inhibition added based on Palatsi et al 2010*/
  proc8 = k_m_c4*x[3]/(K_S_c4+x[3]*(1 + (x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))) / pH_UL_aa))*x[19]*x[3]/(x[3]+x[4]+eps)*inhib[2]*exp(3101*(T_op-T_base)/T_op/T_base);
  proc9 = k_m_c4*x[4]/(K_S_c4+x[4]*(1 + (x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))) / pH_UL_aa))*x[19]*x[4]/(x[3]+x[4]+eps)*inhib[2]*exp(3101*(T_op-T_base)/T_op/T_base);
  proc10 = k_m_pro*x[5]/(K_S_pro+x[5]*(1 + (x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))) / pH_UL_aa))*x[20]*inhib[3]*exp(6706*(T_op-T_base)/T_op/T_base);
    
    //    Proc5-10 used I_ph_aa; equation from Zhu et al (2018). (Equation 11) factor of (1 + [product]/Km) Km is suspended constant
    //    *(1 + (x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))) / 0.0947) HAc
    
  proc11 = k_m_ac*(x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op)))))/(K_S_ac+x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))+pow(x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op)))),2)/pH_UL_ac)*x[21]*(2370*x[18]/(2370*x[18]+x[2]*x[2]))*inhib[4]*exp(8184*(T_op-T_base)/T_op/T_base); /*pH_UL_ac repurposed to represent Haldane inhibition constant K_I_ac_ac*//*added LCFA inhibition - Palatsi et al 2010*/
//    k_m_ac*x[6]/(K_S_ac+x[6])*x[21]*inhib[4];
  proc12 = k_m_h2*x[7]/(K_S_h2+x[7])*x[22]*(2370*x[18]/(2370*x[18]+x[2]*x[2]))*inhib[5]*exp(4803*(T_op-T_base)/T_op/T_base);/*added LCFA inhibition*/
  proc13 = k_dec_Xsu*x[16];
  proc14 = k_dec_Xaa*x[17];
  proc15 = k_dec_Xfa*x[18];
  proc16 = k_dec_Xc4*x[19];
  proc17 = k_dec_Xpro*x[20];
  proc18 = k_dec_Xac*x[21];
  proc19 = k_dec_Xh2*x[22];


  procT8 = kLa*(x[7]-16.0*K_H_h2*p_gas_h2)*pow(1.012,(T_op-T_base)); //
  procT9 = kLa*(x[8]-64.0*K_H_ch4*p_gas_ch4)*pow(1.012,(T_op-T_base)); //
  procT10 = kLa*((x[43])-K_H_co2*p_gas_co2)*pow(1.012,(T_op-T_base));//

  /* Stoichiometry for C and N balance */ 
  stoich1 = -C_xc+f_sI_xc*C_sI+f_ch_xc*C_ch+f_pr_xc*C_pr+f_li_xc*C_li+f_xI_xc*C_xI;
  stoich2 = -C_ch+C_su;
  stoich3 = -C_pr+C_aa;
  stoich4 = -C_li+(1.0-f_fa_li)*C_su+f_fa_li*C_fa;
  stoich5 = -C_su+(1.0-Y_su)*(f_bu_su*C_bu+f_pro_su*C_pro+f_ac_su*C_ac)+Y_su*C_bac;
  stoich6 = -C_aa+(1.0-Y_aa)*(f_va_aa*C_va+f_bu_aa*C_bu+f_pro_aa*C_pro+f_ac_aa*C_ac)+Y_aa*C_bac;
  stoich7 = -C_fa+(1.0-Y_fa)*0.7*C_ac+Y_fa*C_bac;
  stoich8 = -C_va+(1.0-Y_c4)*0.54*C_pro+(1.0-Y_c4)*0.31*C_ac+Y_c4*C_bac;
  stoich9 = -C_bu+(1.0-Y_c4)*0.8*C_ac+Y_c4*C_bac;
  stoich10 = -C_pro+(1.0-Y_pro)*0.57*C_ac+Y_pro*C_bac;
  stoich11 = -C_ac+(1.0-Y_ac)*C_ch4+Y_ac*C_bac;
  stoich12 = (1.0-Y_h2)*C_ch4+Y_h2*C_bac;
  stoich13 = -C_bac+C_xc;

  /*Overall Reaction Rates for State Variables*/
  reac1 = proc2+(1.0-f_fa_li)*proc4-proc5;
  reac2 = proc3-proc6;
  reac3 = f_fa_li*proc4-proc7;
  reac4 = (1.0-Y_aa)*f_va_aa*proc6-proc8;
  reac5 = (1.0-Y_su)*f_bu_su*proc5+(1.0-Y_aa)*f_bu_aa*proc6-proc9;
  reac6 = (1.0-Y_su)*f_pro_su*proc5+(1.0-Y_aa)*f_pro_aa*proc6+(1.0-Y_c4)*0.54*proc8-proc10; /*propionate production = propionate consumption*/
  reac7 = (1.0-Y_su)*f_ac_su*proc5+(1.0-Y_aa)*f_ac_aa*proc6+(1.0-Y_fa)*0.7*proc7+(1.0-Y_c4)*0.31*proc8+(1.0-Y_c4)*0.8*proc9+(1.0-Y_pro)*0.57*proc10-proc11; /*acetate production = acetate consumption*/
  /*reac 8 is not used because hydrogen is solved algebraically  reac8 = (1.0-Y_su)*f_h2_su*proc5+(1.0-Y_aa)*f_h2_aa*proc6+(1.0-Y_fa)*0.3*proc7+(1.0-Y_c4)*0.15*proc8+(1.0-Y_c4)*0.2*proc9+(1.0-Y_pro)*0.43*proc10-proc12-procT8;*/
  reac8 = (1.0-Y_su)*f_h2_su*proc5+(1.0-Y_aa)*f_h2_aa*proc6+(1.0-Y_fa)*0.3*proc7+(1.0-Y_c4)*0.15*proc8+(1.0-Y_c4)*0.2*proc9+(1.0-Y_pro)*0.43*proc10-proc12-procT8; /*H2 production = H2 consumption + mass transfer*/
  reac9 = (1.0-Y_ac)*proc11+(1.0-Y_h2)*proc12-procT9;
  reac10 = -stoich1*proc1-stoich2*proc2-stoich3*proc3-stoich4*proc4-stoich5*proc5-stoich6*proc6-stoich7*proc7-stoich8*proc8-stoich9*proc9-stoich10*proc10-stoich11*proc11-stoich12*proc12-stoich13*proc13-stoich13*proc14-stoich13*proc15-stoich13*proc16-stoich13*proc17-stoich13*proc18-stoich13*proc19-procT10;
  reac11 = (N_xc-f_xI_xc*N_I-f_sI_xc*N_I-f_pr_xc*N_aa)*proc1-Y_su*N_bac*proc5+(N_aa-Y_aa*N_bac)*proc6-Y_fa*N_bac*proc7-Y_c4*N_bac*proc8-Y_c4*N_bac*proc9-Y_pro*N_bac*proc10-Y_ac*N_bac*proc11-Y_h2*N_bac*proc12+(N_bac-N_xc)*(proc13+proc14+proc15+proc16+proc17+proc18+proc19);
  reac12 = f_sI_xc*proc1;
  reac13 = -proc1+proc13+proc14+proc15+proc16+proc17+proc18+proc19;
  reac14 = f_ch_xc*proc1-proc2;
  reac15 = f_pr_xc*proc1-proc3;
  reac16 = f_li_xc*proc1-proc4;
  reac17 = Y_su*proc5-proc13;
  reac18 = Y_aa*proc6-proc14;
  reac19 = Y_fa*proc7-proc15;
  reac20 = Y_c4*proc8+Y_c4*proc9-proc16;
  reac21 = Y_pro*proc10-proc17;
  reac22 = Y_ac*proc11-proc18;
  reac23 = Y_h2*proc12-proc19;
  reac24 = f_xI_xc*proc1;

  q_gas = k_P*(P_gas-P_atm);
  /*if (q_gas < 0)
      q_gas = 0.0;*/

  V_frac=1.0/V_liq;

  /* State Variables: q/V*(Sin - S) + reaction(s) */  
  //f[0] = xdot[0]-V_frac*(influent[26]*(influent[0]-x[0]))-reac1; 		/* Ssu */
  //f[1] = xdot[1]-V_frac*(influent[26]*(influent[1]-x[1]))-reac2; 		/* Saa */
  //f[2] = xdot[2]-V_frac*(influent[26]*(influent[2]-x[2]))-reac3;		/* Sfa */
  //f[3] = xdot[3]-V_frac*(influent[26]*(influent[3]-x[3]))-reac4;  		/* Sva */
  //f[4] = xdot[4]-V_frac*(influent[26]*(influent[4]-x[4]))-reac5;  		/* Sbu */
  //f[5] = xdot[5]-V_frac*(influent[26]*(influent[5]-x[5]))-reac6;  		/* Spro */
  //f[6] = xdot[6]-V_frac*(influent[26]*(influent[6]-x[6]))-reac7;  		/* Sac */
  //f[7] = V_frac*(influent[26]*(influent[7]-x[7]))-reac8;			/* Sh2 */
  //f[8] = xdot[8]-V_frac*(influent[26]*(influent[8]-x[8]))-reac9;     	/* Sch4 */
  //f[9] = xdot[9]-V_frac*(influent[26]*(influent[9]-x[9]))-reac10;   	/* SIC */
  //f[10] = xdot[10]-V_frac*(influent[26]*(influent[10]-x[10]))-reac11; 	/* SIN */
  //f[11] = xdot[11]-V_frac*(influent[26]*(influent[11]-x[11]))-reac12;  	/* SI */
  //f[12] = xdot[12]-V_frac*(influent[26]*(influent[12]-x[12]))-reac13; 	/* Xxc */
  //f[13] = xdot[13]-V_frac*(influent[26]*(influent[13]-x[13]))-reac14; 	/* Xch */
  //f[14] = xdot[14]-V_frac*(influent[26]*(influent[14]-x[14]))-reac15; 	/* Xpr */
  //f[15] = xdot[15]-V_frac*(influent[26]*(influent[15]-x[15]))-reac16; 	/* Xli */
  //f[16] = xdot[16]-V_frac*(influent[26]*(influent[16]-x[16]))-reac17; 	/* Xsu */
  //f[17] = xdot[17]-V_frac*(influent[26]*(influent[17]-x[17]))-reac18; 	/* Xaa */
  //f[18] = xdot[18]-V_frac*(influent[26]*(influent[18]-x[18]))-reac19; 	/* Xfa */
  //f[19] = xdot[19]-V_frac*(influent[26]*(influent[19]-x[19]))-reac20; 	/* Xc4 */
  //f[20] = xdot[20]-V_frac*(influent[26]*(influent[20]-x[20]))-reac21; 	/* Xpro */
  //f[21] = xdot[21]-V_frac*(influent[26]*(influent[21]-x[21]))-reac22;	/* Xac */
  //f[22] = xdot[22]-V_frac*(influent[26]*(influent[22]-x[22]))-reac23; 	/* Xh2 */
  //f[23] = xdot[23]-V_frac*(influent[26]*(influent[23]-x[23]))-reac24; 	/* XI */
  //f[24] = xdot[24]-V_frac*(influent[26]*(influent[24]-x[24])); 			/* Scat+ */

/* State Variables: q/V*(Sin - S) + reaction(s) */  
  f[0] = xdot[0]-V_frac*(influent[26]*(influent[0]-x[0]))-reac1; 		/* Ssu */
  f[1] = xdot[1]-V_frac*(influent[26]*(influent[1]-x[1]))-reac2; 		/* Saa */
  f[2] = xdot[2]-V_frac*(influent[26]*(influent[2]-x[2]))-reac3;		/* Sfa */
  f[3] = xdot[3]-V_frac*(influent[26]*(influent[3]-x[3]))-reac4;  		/* Sva */
  f[4] = xdot[4]-V_frac*(influent[26]*(influent[4]-x[4]))-reac5;  		/* Sbu */
  f[5] = xdot[5]-V_frac*(influent[26]*(influent[5]-x[5]))-reac6;  		/* Spro */
  f[6] = xdot[6]-V_frac*(influent[26]*(influent[6]-x[6]))-reac7;  		/* Sac */
  f[7] = V_frac*(influent[26]*(influent[7]-x[7]))-reac8;			/* Sh2 */
  f[8] = xdot[8]-V_frac*(influent[26]*(influent[8]-x[8]))-reac9;     	/* Sch4 */
  f[9] = xdot[9]-V_frac*(influent[26]*(influent[9]-x[9]))-reac10;   	/* SIC */
  f[10] = xdot[10]-V_frac*(influent[26]*(influent[10]-x[10]))-reac11; 	/* SIN */
  f[11] = xdot[11]-V_frac*(influent[26]*(influent[11]-x[11]))-reac12;  	/* SI */
  f[12] = xdot[12]-( V_frac*influent[26]*influent[12]-x[12]/(V_liq/influent[26] + t_resx))-reac13; 	/* Xxc */
  f[13] = xdot[13]-( V_frac*influent[26]*influent[13]-x[13]/(V_liq/influent[26] + t_resx))-reac14; 	/* Xch */
  f[14] = xdot[14]-( V_frac*influent[26]*influent[14]-x[14]/(V_liq/influent[26] + t_resx))-reac15; 	/* Xpr */
  f[15] = xdot[15]-( V_frac*influent[26]*influent[15]-x[15]/(V_liq/influent[26] + t_resx))-reac16; 	/* Xli */
  f[16] = xdot[16]-( V_frac*influent[26]*influent[16]-x[16]/(V_liq/influent[26] + t_resx))-reac17; 	/* Xsu */
  f[17] = xdot[17]-( V_frac*influent[26]*influent[17]-x[17]/(V_liq/influent[26] + t_resx))-reac18; 	/* Xaa */
  f[18] = xdot[18]-( V_frac*influent[26]*influent[18]-x[18]/(V_liq/influent[26] + t_resx))-reac19; 	/* Xfa */
  f[19] = xdot[19]-( V_frac*influent[26]*influent[19]-x[19]/(V_liq/influent[26] + t_resx))-reac20; 	/* Xc4 */
  f[20] = xdot[20]-( V_frac*influent[26]*influent[20]-x[20]/(V_liq/influent[26] + t_resx))-reac21; 	/* Xpro */
  f[21] = xdot[21]-( V_frac*influent[26]*influent[21]-x[21]/(V_liq/influent[26] + t_resx))-reac22;	/* Xac */
  f[22] = xdot[22]-( V_frac*influent[26]*influent[22]-x[22]/(V_liq/influent[26] + t_resx))-reac23; 	/* Xh2 */
  f[23] = xdot[23]-( V_frac*influent[26]*influent[23]-x[23]/(V_liq/influent[26] + t_resx))-reac24; 	/* XI */
  f[24] = xdot[24]-V_frac*(influent[26]*(influent[24]-x[24])); 			                                  /* Scat+ */
  if (ctx->set_Cat_mass) {
    f[24] = f[24]+ctx->Cat_mass; 			/* Scat+ */
  }
  f[25] = xdot[25]-V_frac*(influent[26]*(influent[25]-x[25])); 			/* San- */


  f[26] = x[26]-K_a_va*x[3]/(K_a_va+x[42]);   /* Sva- */
  f[27] = x[27]-K_a_bu*x[4]/(K_a_bu+x[42]);   /* Sbu- */
  f[28] = x[28]-K_a_pro*x[5]/(K_a_pro+x[42]); /* Spro- */
  f[29] = x[29]-K_a_ac*x[6]/(K_a_ac+x[42]);   /* Sac- */
  f[30] = x[30]-K_a_co2*x[9]/(K_a_co2+x[42]); /* SHCO3- */
  f[31] = x[31]-K_a_IN*x[10]/(K_a_IN+x[42]);  /* SNH3 */

  f[32] = xdot[32]+x[32]*q_gas/V_gas-procT8*V_liq/V_gas;  /* Sgas,H2 */
  f[33] = xdot[33]+x[33]*q_gas/V_gas-procT9*V_liq/V_gas;  /* Sgas,CH4 */
  f[34] = xdot[34]+x[34]*q_gas/V_gas-procT10*V_liq/V_gas; /* Sgas,CO2 */
        
  f[35] = xdot[35]; 											/* Flow is constant*/
  f[36] = xdot[36]; 											/* Temp is constant*/

  /* Dummy states*/
  f[37] = xdot[37];
  f[38] = xdot[38];
  f[39] = xdot[39];
  f[40] = xdot[40];
  f[41] = xdot[41];

  f[42] = x[24]+(x[10]-x[31])+x[42]-x[30]-x[29]/64.0-x[28]/112.0-x[27]/160.0-x[26]/208.0-K_w/x[42]-x[25]; /* SH+ */

  f[43] = x[43] - (x[9] - x[30]);				 /* SCO2 */
  f[44] = x[44] - (x[10] - x[31]);			 /* SNH4+ */
  
  proc11_max = k_m_ac*x[21]*inhib[4];
  ctx->rwork = proc11_max/reac7;  	     /* acetate capacity number (ACN) in (kg COD/m3/d)/(kg COD/m3/d) */ 

  ierr = VecRestoreArrayRead(ctx->influent,&influent);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Udot,&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  'Active' ADOL-C annotated version, marking dependence upon u.
*/
PetscErrorCode IFunctionActive1(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,AppCtx *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *f;
  const PetscScalar *x,*xdot,*params,*influent; 

  adouble           f_a[45]; /* 'active' double for dependent variables */
  adouble           x_a[45]; /* 'active' double for independent variables */

  adouble f_sI_xc, f_xI_xc, f_ch_xc, f_pr_xc, f_li_xc, N_xc, N_I, N_aa, C_xc, C_sI, C_ch, C_pr, C_li, C_xI, C_su, C_aa, f_fa_li, C_fa, f_h2_su, f_bu_su, f_pro_su, f_ac_su, N_bac, C_bu, C_pro, C_ac, C_bac, Y_su, f_h2_aa, f_va_aa, f_bu_aa, f_pro_aa, f_ac_aa, C_va, Y_aa, Y_fa, Y_c4, Y_pro, C_ch4, Y_ac, Y_h2;
  adouble k_dis, k_hyd_ch, k_hyd_pr, k_hyd_li, K_S_IN, k_m_su, K_S_su, pH_UL_aa, pH_LL_aa;
  adouble k_m_aa, K_S_aa, k_m_fa, K_S_fa, K_Ih2_fa, k_m_c4, K_S_c4, K_Ih2_c4, k_m_pro, K_S_pro;
  adouble K_Ih2_pro, k_m_ac, K_S_ac, K_I_nh3, pH_UL_ac, pH_LL_ac, k_m_h2, K_S_h2, pH_UL_h2, pH_LL_h2;
  adouble k_dec_Xsu, k_dec_Xaa, k_dec_Xfa, k_dec_Xc4, k_dec_Xpro, k_dec_Xac, k_dec_Xh2;
  adouble R, T_base, T_op;
  adouble K_H_h2_base, factor; 
  adouble P_atm, p_gas_h2o, P_gas, k_P, kLa, K_H_co2, K_H_ch4, K_H_h2;
  adouble V_liq, V_gas, t_resx, V_frac;
  adouble eps, pH_op, S_H_ion;
  adouble proc1, proc2, proc3, proc4, proc5, proc6, proc7, proc8, proc9, proc10, proc11, proc12, proc13, proc14, proc15, proc16, proc17, proc18, proc19, procT8, procT9, procT10;
  adouble I_pH_aa, I_pH_ac, I_pH_h2, I_IN_lim, I_h2_fa, I_h2_c4, I_h2_pro, I_nh3;
  adouble reac1, reac2, reac3, reac4, reac5, reac6, reac7, reac8, reac9, reac10, reac11, reac12, reac13, reac14, reac15, reac16, reac17, reac18, reac19, reac20, reac21, reac22, reac23, reac24, stoich1, stoich2, stoich3, stoich4, stoich5, stoich6, stoich7, stoich8, stoich9, stoich10, stoich11, stoich12, stoich13;
  adouble inhib[6]; /*declare arrays */
  adouble p_gas_h2, p_gas_ch4, p_gas_co2, q_gas;
  adouble pHLim_aa, pHLim_ac, pHLim_h2, n_aa, n_ac, n_h2;
  // PetscInt i;
  adouble K_w, pK_w_base, K_a_va, pK_a_va_base, K_a_bu, pK_a_bu_base, K_a_pro, pK_a_pro_base, K_a_ac, pK_a_ac_base, K_a_co2, pK_a_co2_base, K_a_IN, pK_a_IN_base;

  eps = 0.0;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(ctx->params,&params);CHKERRQ(ierr);
  f_sI_xc = params[0];
  f_xI_xc = params[1];
  f_ch_xc = params[2];
  f_pr_xc = params[3];
  f_li_xc = params[4];
  N_xc = params[5];
  N_I = params[6];
  N_aa = params[7];
  C_xc = params[8];
  C_sI = params[9];
  C_ch = params[10];
  C_pr = params[11];
  C_li = params[12];
  C_xI = params[13];
  C_su = params[14];
  C_aa = params[15];
  f_fa_li = params[16];
  C_fa = params[17];
  f_h2_su = params[18];
  f_bu_su = params[19];
  f_pro_su = params[20];
  f_ac_su = params[21];
  N_bac = params[22];
  C_bu = params[23];
  C_pro = params[24];
  C_ac = params[25];
  C_bac = params[26];
  Y_su = params[27];
  f_h2_aa = params[28];
  f_va_aa = params[29];
  f_bu_aa = params[30];
  f_pro_aa = params[31];
  f_ac_aa = params[32];
  C_va = params[33];
  Y_aa = params[34];
  Y_fa = params[35];
  Y_c4 = params[36];
  Y_pro = params[37];
  C_ch4 = params[38];
  Y_ac = params[39];
  Y_h2 = params[40];
  k_dis = params[41];
  k_hyd_ch = params[42];
  k_hyd_pr = params[43];
  k_hyd_li = params[44];
  K_S_IN = params[45];
  k_m_su = params[46];
  K_S_su = params[47];
  pH_UL_aa = params[48];
  pH_LL_aa = params[49];
  k_m_aa = params[50];
  K_S_aa = params[51];
  k_m_fa = params[52];
  K_S_fa = params[53];
  K_Ih2_fa = params[54];
  k_m_c4 = params[55];
  K_S_c4 = params[56];
  K_Ih2_c4 = params[57];
  k_m_pro = params[58];
  K_S_pro = params[59];
  K_Ih2_pro = params[60];
  k_m_ac = params[61];
  K_S_ac = params[62];
  K_I_nh3 = params[63];
  pH_UL_ac = params[64];
  pH_LL_ac = params[65];
  k_m_h2 = params[66];
  K_S_h2 = params[67];
  pH_UL_h2 = params[68];
  pH_LL_h2 = params[69];
  k_dec_Xsu = params[70];
  k_dec_Xaa = params[71];
  k_dec_Xfa = params[72];
  k_dec_Xc4 = params[73];
  k_dec_Xpro = params[74];
  k_dec_Xac = params[75];
  k_dec_Xh2 = params[76];
  R = params[77];
  T_base = params[78];
  T_op = params[79];
  pK_w_base = params[80];
  pK_a_va_base = params[81];
  pK_a_bu_base = params[82];
  pK_a_pro_base = params[83];
  pK_a_ac_base = params[84];
  pK_a_co2_base = params[85];
  pK_a_IN_base = params[86];

  P_atm = params[93];
  kLa = params[94];
  K_H_h2_base = params[98];
  k_P = params[99];

  ierr = VecRestoreArrayRead(ctx->params,&params);CHKERRQ(ierr);

  V_liq = ctx->V[0];
  V_gas = ctx->V[1];


  /* Set the residual */
  ierr = VecGetArrayRead(ctx->influent,&influent);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  /* Start of active section */
  trace_on(1);
  x_a[0] <<= x[0];
  x_a[1] <<= x[1];
  x_a[2] <<= x[2];
  x_a[3] <<= x[3];
  x_a[4] <<= x[4];
  x_a[5] <<= x[5];
  x_a[6] <<= x[6];
  x_a[7] <<= x[7];
  x_a[8] <<= x[8];
  x_a[9] <<= x[9];
  x_a[10] <<= x[10];
  x_a[11] <<= x[11];
  x_a[12] <<= x[12];
  x_a[13] <<= x[13];
  x_a[14] <<= x[14];
  x_a[15] <<= x[15];
  x_a[16] <<= x[16];
  x_a[17] <<= x[17];
  x_a[18] <<= x[18];
  x_a[19] <<= x[19];
  x_a[20] <<= x[20];
  x_a[21] <<= x[21];
  x_a[22] <<= x[22];
  x_a[23] <<= x[23];
  x_a[24] <<= x[24];
  x_a[25] <<= x[25];
  x_a[26] <<= x[26];
  x_a[27] <<= x[27];
  x_a[28] <<= x[28];
  x_a[29] <<= x[29];
  x_a[30] <<= x[30];
  x_a[31] <<= x[31];
  x_a[32] <<= x[32];
  x_a[33] <<= x[33];
  x_a[34] <<= x[34];
  x_a[35] <<= x[35];
  x_a[36] <<= x[36];
  x_a[37] <<= x[37];
  x_a[38] <<= x[38];
  x_a[39] <<= x[39];
  x_a[40] <<= x[40];
  x_a[41] <<= x[41];
  x_a[42] <<= x[42];
  x_a[43] <<= x[43];
  x_a[44] <<= x[44];
  /* Mark independence */


  factor = (1.0/T_base - 1.0/T_op)/(100.0*R);
  K_a_va = pow(10,-pK_a_va_base);
  K_a_bu = pow(10,-pK_a_bu_base);
  K_a_pro = pow(10,-pK_a_pro_base);
  K_w = pow(10,-pK_w_base)*exp(55700.0*factor); /* T adjustment for K_w */
  K_a_ac = pow(10,-pK_a_ac_base)*exp(-4600.0*factor);  /* T adjustment */
  K_a_co2 = pow(10,-pK_a_co2_base)*exp(7600.0*factor); /* T adjustment for K_a_co2 */
  K_a_IN = pow(10,-pK_a_IN_base)*exp(51800.0*factor);  /* T adjustment for K_a_IN */


  K_H_h2 = 1.0/pow(10,(-187.04/T_op+5.473))*55.6/1.01325;     /* atm*L/mol; modified van't Hoff-Arrhenius relationship; MetCalf & Eddy, p68; conversions 55.6 mole H2O/L; 1 atm = 1.01325 */
  K_H_ch4 = 1.0/pow(10,(-675.74/T_op+6.880))*55.6/1.01325; /* atm*L/mol; modified van't Hoff-Arrhenius relationship; MetCalf & Eddy, p68 */
  K_H_co2 = 1.0/pow(10,(-1012.40/T_op+6.606))*55.6/1.01325; /* atm*L/mol; modified van't Hoff-Arrhenius relationship; MetCalf & Eddy, p68 */

  p_gas_h2o = pow(10, (5.20389-1733.926/(T_op-39.485)));  /* Antoine equation for water pressure for temp range 31C-60C */ 

  p_gas_h2 = x_a[32]*R*T_op/16.0;
  p_gas_ch4 = x_a[33]*R*T_op/64.0;
  p_gas_co2 = x_a[34]*R*T_op;
  P_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o;

  S_H_ion = x_a[42];
  pH_op = -log10(x_a[42]);   /* pH */


  /* Inhibition */
  /* Hill function on SH+ used within BSM2, ADM1 Workshop, Copenhagen 2005 */
  /*Hill function is replaced by other inhibition terms, assign value to the parameters in Hill function to avoid error*/
//  pHLim_aa = pow(10,(-(pH_UL_aa + pH_LL_aa)/2.0));
//  pHLim_ac = pow(10,(-(pH_UL_ac + pH_LL_ac)/2.0));
//  pHLim_h2 = pow(10,(-(pH_UL_h2 + pH_LL_h2)/2.0));
//  n_aa=3.0/(pH_UL_aa-pH_LL_aa);
//  n_ac=3.0/(pH_UL_ac-pH_LL_ac);
//  n_h2=3.0/(pH_UL_h2-pH_LL_h2);
//  I_pH_aa = pow(pHLim_aa,n_aa)/(pow(S_H_ion,n_aa)+pow(pHLim_aa,n_aa));
//  I_pH_ac = pow(pHLim_ac,n_ac)/(pow(S_H_ion,n_ac)+pow(pHLim_ac,n_ac));
//  I_pH_h2 = pow(pHLim_h2,n_h2)/(pow(S_H_ion,n_h2)+pow(pHLim_h2,n_h2));
  pHLim_aa = 1;
  pHLim_ac = 1;
  pHLim_h2 = 1;
  n_aa=1;
  n_ac=1;
  n_h2=1;
  I_pH_aa = 1;
  I_pH_ac = 1;

  /*New inhibition terms used*/
  I_pH_h2 = (pH_UL_h2) / ( pH_UL_h2 + x_a[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))*60/(64*1000) );


  I_IN_lim = 1.0/(1.0+K_S_IN/x_a[10]);
  I_h2_fa = 1.0/(1.0+x_a[7]/K_Ih2_fa);//
  I_h2_c4 = 1.0/(1.0+x_a[7]/K_Ih2_c4);//
  I_h2_pro = 1.0/(1.0+x_a[7]/K_Ih2_pro);//
  I_nh3 = 1.0/(1.0+x_a[31]/K_I_nh3);//

  inhib[0] = I_IN_lim;
//    I_pH_aa*I_IN_lim;
  inhib[1] = inhib[0]*I_h2_fa;
  inhib[2] = inhib[0]*I_h2_c4;
  inhib[3] = inhib[0]*I_h2_pro;
  inhib[4] = I_IN_lim*I_nh3;
//    I_pH_ac*I_IN_lim*I_nh3;
  inhib[5] = I_pH_h2*I_IN_lim;

  /* Process Rates*/  
  proc1 = k_dis*x_a[12]*exp(3737*(T_op-T_base)/T_op/T_base);
  proc2 = k_hyd_ch*x_a[13]*exp(3737*(T_op-T_base)/T_op/T_base);
  proc3 = k_hyd_pr*x_a[14]*exp(3737*(T_op-T_base)/T_op/T_base);
  proc4 = k_hyd_li*x_a[15]*exp(3737*(T_op-T_base)/T_op/T_base);
    
  proc5 = k_m_su*x_a[0]/(K_S_su+x_a[0]*(1 + (x_a[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))) / pH_UL_aa))*x_a[16]*inhib[0]*exp(3101*(T_op-T_base)/T_op/T_base);
  proc6 = k_m_aa*x_a[1]/(K_S_aa+x_a[1]*(1 + (x_a[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))) / pH_UL_aa))*x_a[17]*inhib[0]*exp(3101*(T_op-T_base)/T_op/T_base);
  proc7 = k_m_fa*x_a[2]*2370*x_a[18]/(K_S_fa*2370*x_a[18]+x_a[2]*(1 + (x_a[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))) / pH_UL_aa)*2370*x_a[18]+x_a[2]*x_a[2]*x_a[2])*x_a[18]*inhib[1]*exp(3101*(T_op-T_base)/T_op/T_base);/*added LCFA inhibition from Palatsi et al. 2010*/
  proc8 = k_m_c4*x_a[3]/(K_S_c4+x_a[3]*(1 + (x_a[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))) / pH_UL_aa))*x_a[19]*x_a[3]/(x_a[3]+x_a[4]+eps)*inhib[2]*exp(3101*(T_op-T_base)/T_op/T_base);
  proc9 = k_m_c4*x_a[4]/(K_S_c4+x_a[4]*(1 + (x_a[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))) / pH_UL_aa))*x_a[19]*x_a[4]/(x_a[3]+x_a[4]+eps)*inhib[2]*exp(3101*(T_op-T_base)/T_op/T_base);
  proc10 = k_m_pro*x_a[5]/(K_S_pro+x_a[5]*(1 + (x_a[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))) / pH_UL_aa))*x_a[20]*inhib[3]*exp(6706*(T_op-T_base)/T_op/T_base);
    
    //    Proc5-10 used I_ph_aa; equation from Zhu et al (2018). (Equation 11) factor of (1 + [product]/Km) Km is suspended constant
    //    *(1 + (x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))) / 0.0947) HAc
    
  proc11 = k_m_ac*(x_a[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op)))))/(K_S_ac+x_a[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))+pow(x_a[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op)))),2)/pH_UL_ac)*x_a[21]*(2370*x_a[18]/(2370*x_a[18]+x_a[2]*x_a[2]))*inhib[4]*exp(8184*(T_op-T_base)/T_op/T_base);/*added LCFA inhibition from Palatsi et al 2010*/
//    k_m_ac*x_a[6]/(K_S_ac+x_a[6])*x_a[21]*inhib[4];
  proc12 = k_m_h2*x_a[7]/(K_S_h2+x_a[7])*x_a[22]*(2370*x_a[18]/(2370*x_a[18]+x_a[2]*x_a[2]))*inhib[5]*exp(4803*(T_op-T_base)/T_op/T_base);//
  proc13 = k_dec_Xsu*x_a[16];
  proc14 = k_dec_Xaa*x_a[17];
  proc15 = k_dec_Xfa*x_a[18];
  proc16 = k_dec_Xc4*x_a[19];
  proc17 = k_dec_Xpro*x_a[20];
  proc18 = k_dec_Xac*x_a[21];
  proc19 = k_dec_Xh2*x_a[22];


  procT8 = kLa*(x_a[7]-16.0*K_H_h2*p_gas_h2)*pow(1.012,(T_op-T_base)); //
  procT9 = kLa*(x_a[8]-64.0*K_H_ch4*p_gas_ch4)*pow(1.012,(T_op-T_base)); //
  procT10 = kLa*((x_a[43])-K_H_co2*p_gas_co2)*pow(1.012,(T_op-T_base));//

  /* Stoichiometry for C and N balance */ 
  stoich1 = -C_xc+f_sI_xc*C_sI+f_ch_xc*C_ch+f_pr_xc*C_pr+f_li_xc*C_li+f_xI_xc*C_xI;
  stoich2 = -C_ch+C_su;
  stoich3 = -C_pr+C_aa;
  stoich4 = -C_li+(1.0-f_fa_li)*C_su+f_fa_li*C_fa;
  stoich5 = -C_su+(1.0-Y_su)*(f_bu_su*C_bu+f_pro_su*C_pro+f_ac_su*C_ac)+Y_su*C_bac;
  stoich6 = -C_aa+(1.0-Y_aa)*(f_va_aa*C_va+f_bu_aa*C_bu+f_pro_aa*C_pro+f_ac_aa*C_ac)+Y_aa*C_bac;
  stoich7 = -C_fa+(1.0-Y_fa)*0.7*C_ac+Y_fa*C_bac;
  stoich8 = -C_va+(1.0-Y_c4)*0.54*C_pro+(1.0-Y_c4)*0.31*C_ac+Y_c4*C_bac;
  stoich9 = -C_bu+(1.0-Y_c4)*0.8*C_ac+Y_c4*C_bac;
  stoich10 = -C_pro+(1.0-Y_pro)*0.57*C_ac+Y_pro*C_bac;
  stoich11 = -C_ac+(1.0-Y_ac)*C_ch4+Y_ac*C_bac;
  stoich12 = (1.0-Y_h2)*C_ch4+Y_h2*C_bac;
  stoich13 = -C_bac+C_xc;

  /*Overall Reaction Rates for State Variables*/
  reac1 = proc2+(1.0-f_fa_li)*proc4-proc5;
  reac2 = proc3-proc6;
  reac3 = f_fa_li*proc4-proc7;
  reac4 = (1.0-Y_aa)*f_va_aa*proc6-proc8;
  reac5 = (1.0-Y_su)*f_bu_su*proc5+(1.0-Y_aa)*f_bu_aa*proc6-proc9;
  reac6 = (1.0-Y_su)*f_pro_su*proc5+(1.0-Y_aa)*f_pro_aa*proc6+(1.0-Y_c4)*0.54*proc8-proc10;
  reac7 = (1.0-Y_su)*f_ac_su*proc5+(1.0-Y_aa)*f_ac_aa*proc6+(1.0-Y_fa)*0.7*proc7+(1.0-Y_c4)*0.31*proc8+(1.0-Y_c4)*0.8*proc9+(1.0-Y_pro)*0.57*proc10-proc11;
  /*reac 8 is not used because hydrogen is solved algebraically  reac8 = (1.0-Y_su)*f_h2_su*proc5+(1.0-Y_aa)*f_h2_aa*proc6+(1.0-Y_fa)*0.3*proc7+(1.0-Y_c4)*0.15*proc8+(1.0-Y_c4)*0.2*proc9+(1.0-Y_pro)*0.43*proc10-proc12-procT8;*/
  reac8 = (1.0-Y_su)*f_h2_su*proc5+(1.0-Y_aa)*f_h2_aa*proc6+(1.0-Y_fa)*0.3*proc7+(1.0-Y_c4)*0.15*proc8+(1.0-Y_c4)*0.2*proc9+(1.0-Y_pro)*0.43*proc10-proc12-procT8;
  reac9 = (1.0-Y_ac)*proc11+(1.0-Y_h2)*proc12-procT9;
  reac10 = -stoich1*proc1-stoich2*proc2-stoich3*proc3-stoich4*proc4-stoich5*proc5-stoich6*proc6-stoich7*proc7-stoich8*proc8-stoich9*proc9-stoich10*proc10-stoich11*proc11-stoich12*proc12-stoich13*proc13-stoich13*proc14-stoich13*proc15-stoich13*proc16-stoich13*proc17-stoich13*proc18-stoich13*proc19-procT10;
  reac11 = (N_xc-f_xI_xc*N_I-f_sI_xc*N_I-f_pr_xc*N_aa)*proc1-Y_su*N_bac*proc5+(N_aa-Y_aa*N_bac)*proc6-Y_fa*N_bac*proc7-Y_c4*N_bac*proc8-Y_c4*N_bac*proc9-Y_pro*N_bac*proc10-Y_ac*N_bac*proc11-Y_h2*N_bac*proc12+(N_bac-N_xc)*(proc13+proc14+proc15+proc16+proc17+proc18+proc19);
  reac12 = f_sI_xc*proc1;
  reac13 = -proc1+proc13+proc14+proc15+proc16+proc17+proc18+proc19;
  reac14 = f_ch_xc*proc1-proc2;
  reac15 = f_pr_xc*proc1-proc3;
  reac16 = f_li_xc*proc1-proc4;
  reac17 = Y_su*proc5-proc13;
  reac18 = Y_aa*proc6-proc14;
  reac19 = Y_fa*proc7-proc15;
  reac20 = Y_c4*proc8+Y_c4*proc9-proc16;
  reac21 = Y_pro*proc10-proc17;
  reac22 = Y_ac*proc11-proc18;
  reac23 = Y_h2*proc12-proc19;
  reac24 = f_xI_xc*proc1;

  q_gas = k_P*(P_gas-P_atm);
  /*if (q_gas < 0)
      q_gas = 0.0;*/

  V_frac=1.0/V_liq;

  /* State Variables: q/V*(Sin - S) + reaction(s) */  
  //f_a[0] = xdot[0]-1.0/V_liq*(influent[26]*(influent[0]-x_a[0]))-reac1; 	  	/* Ssu */
  //f_a[1] = xdot[1]-1.0/V_liq*(influent[26]*(influent[1]-x_a[1]))-reac2; 	  	/* Saa */
  //f_a[2] = xdot[2]-1.0/V_liq*(influent[26]*(influent[2]-x_a[2]))-reac3;		    /* Sfa */
  //f_a[3] = xdot[3]-1.0/V_liq*(influent[26]*(influent[3]-x_a[3]))-reac4;  		  /* Sva */
  //f_a[4] = xdot[4]-1.0/V_liq*(influent[26]*(influent[4]-x_a[4]))-reac5;  		  /* Sbu */
  //f_a[5] = xdot[5]-1.0/V_liq*(influent[26]*(influent[5]-x_a[5]))-reac6;  	  	/* Spro */
  //f_a[6] = xdot[6]-1.0/V_liq*(influent[26]*(influent[6]-x_a[6]))-reac7;  	  	/* Sac */
  //f_a[7] = 1.0/V_liq*(influent[26]*(influent[7]-x_a[7]))-reac8;			          /* Sh2 */
  //f_a[8] = xdot[8]-1.0/V_liq*(influent[26]*(influent[8]-x_a[8]))-reac9;     	/* Sch4 */
  //f_a[9] = xdot[9]-1.0/V_liq*(influent[26]*(influent[9]-x_a[9]))-reac10;   	  /* SIC */
  //f_a[10] = xdot[10]-1.0/V_liq*(influent[26]*(influent[10]-x_a[10]))-reac11; 	/* SIN */
  //f_a[11] = xdot[11]-1.0/V_liq*(influent[26]*(influent[11]-x_a[11]))-reac12; 	/* SI */
  //f_a[12] = xdot[12]-1.0/V_liq*(influent[26]*(influent[12]-x_a[12]))-reac13; 	/* Xxc */
  //f_a[13] = xdot[13]-1.0/V_liq*(influent[26]*(influent[13]-x_a[13]))-reac14; 	/* Xch */
  //f_a[14] = xdot[14]-1.0/V_liq*(influent[26]*(influent[14]-x_a[14]))-reac15; 	/* Xpr */
  //f_a[15] = xdot[15]-1.0/V_liq*(influent[26]*(influent[15]-x_a[15]))-reac16; 	/* Xli */
  //f_a[16] = xdot[16]-1.0/V_liq*(influent[26]*(influent[16]-x_a[16]))-reac17; 	/* Xsu */
  //f_a[17] = xdot[17]-1.0/V_liq*(influent[26]*(influent[17]-x_a[17]))-reac18; 	/* Xaa */
  //f_a[18] = xdot[18]-1.0/V_liq*(influent[26]*(influent[18]-x_a[18]))-reac19; 	/* Xfa */
  //f_a[19] = xdot[19]-1.0/V_liq*(influent[26]*(influent[19]-x_a[19]))-reac20; 	/* Xc4 */
  //f_a[20] = xdot[20]-1.0/V_liq*(influent[26]*(influent[20]-x_a[20]))-reac21; 	/* Xpro */
  //f_a[21] = xdot[21]-1.0/V_liq*(influent[26]*(influent[21]-x_a[21]))-reac22;	/* Xac */
  //f_a[22] = xdot[22]-1.0/V_liq*(influent[26]*(influent[22]-x_a[22]))-reac23; 	/* Xh2 */
  //f_a[23] = xdot[23]-1.0/V_liq*(influent[26]*(influent[23]-x_a[23]))-reac24; 	/* XI */
  //f_a[24] = xdot[24]-1.0/V_liq*(influent[26]*(influent[24]-x_a[24])); 		  	/* Scat+ */

  /* State Variables: q/V*(Sin - S) + reaction(s) */  
  f_a[0] = xdot[0]-V_frac*(influent[26]*(influent[0]-x_a[0]))-reac1; 		/* Ssu */
  f_a[1] = xdot[1]-V_frac*(influent[26]*(influent[1]-x_a[1]))-reac2; 		/* Saa */
  f_a[2] = xdot[2]-V_frac*(influent[26]*(influent[2]-x_a[2]))-reac3;		/* Sfa */
  f_a[3] = xdot[3]-V_frac*(influent[26]*(influent[3]-x_a[3]))-reac4;  		/* Sva */
  f_a[4] = xdot[4]-V_frac*(influent[26]*(influent[4]-x_a[4]))-reac5;  		/* Sbu */
  f_a[5] = xdot[5]-V_frac*(influent[26]*(influent[5]-x_a[5]))-reac6;  		/* Spro */
  f_a[6] = xdot[6]-V_frac*(influent[26]*(influent[6]-x_a[6]))-reac7;  		/* Sac */
  f_a[7] = V_frac*(influent[26]*(influent[7]-x_a[7]))-reac8;			/* Sh2 */
  f_a[8] = xdot[8]-V_frac*(influent[26]*(influent[8]-x_a[8]))-reac9;     	/* Sch4 */
  f_a[9] = xdot[9]-V_frac*(influent[26]*(influent[9]-x_a[9]))-reac10;   	/* SIC */
  f_a[10] = xdot[10]-V_frac*(influent[26]*(influent[10]-x_a[10]))-reac11; 	/* SIN */
  f_a[11] = xdot[11]-V_frac*(influent[26]*(influent[11]-x_a[11]))-reac12;  	/* SI */
  f_a[12] = xdot[12]-( V_frac*influent[26]*influent[12]-x_a[12]/(V_liq/influent[26] + t_resx))-reac13; 	/* Xxc */
  f_a[13] = xdot[13]-( V_frac*influent[26]*influent[13]-x_a[13]/(V_liq/influent[26] + t_resx))-reac14; 	/* Xch */
  f_a[14] = xdot[14]-( V_frac*influent[26]*influent[14]-x_a[14]/(V_liq/influent[26] + t_resx))-reac15; 	/* Xpr */
  f_a[15] = xdot[15]-( V_frac*influent[26]*influent[15]-x_a[15]/(V_liq/influent[26] + t_resx))-reac16; 	/* Xli */
  f_a[16] = xdot[16]-( V_frac*influent[26]*influent[16]-x_a[16]/(V_liq/influent[26] + t_resx))-reac17; 	/* Xsu */
  f_a[17] = xdot[17]-( V_frac*influent[26]*influent[17]-x_a[17]/(V_liq/influent[26] + t_resx))-reac18; 	/* Xaa */
  f_a[18] = xdot[18]-( V_frac*influent[26]*influent[18]-x_a[18]/(V_liq/influent[26] + t_resx))-reac19; 	/* Xfa */
  f_a[19] = xdot[19]-( V_frac*influent[26]*influent[19]-x_a[19]/(V_liq/influent[26] + t_resx))-reac20; 	/* Xc4 */
  f_a[20] = xdot[20]-( V_frac*influent[26]*influent[20]-x_a[20]/(V_liq/influent[26] + t_resx))-reac21; 	/* Xpro */
  f_a[21] = xdot[21]-( V_frac*influent[26]*influent[21]-x_a[21]/(V_liq/influent[26] + t_resx))-reac22;	/* Xac */
  f_a[22] = xdot[22]-( V_frac*influent[26]*influent[22]-x_a[22]/(V_liq/influent[26] + t_resx))-reac23; 	/* Xh2 */
  f_a[23] = xdot[23]-( V_frac*influent[26]*influent[23]-x_a[23]/(V_liq/influent[26] + t_resx))-reac24; 	/* XI */
  f_a[24] = xdot[24]-V_frac*(influent[26]*(influent[24]-x_a[24])); 

  if (ctx->set_Cat_mass) {
    f_a[24] = f_a[24]+ctx->Cat_mass; 			/* Scat+ */
  }  
  f_a[25] = xdot[25]-V_frac*(influent[26]*(influent[25]-x_a[25])); 		  	/* San- */

  f_a[26] = x_a[26]-K_a_va*x_a[3]/(K_a_va+x_a[42]);   /* Sva- */
  f_a[27] = x_a[27]-K_a_bu*x_a[4]/(K_a_bu+x_a[42]);   /* Sbu- */
  f_a[28] = x_a[28]-K_a_pro*x_a[5]/(K_a_pro+x_a[42]); /* Spro- */
  f_a[29] = x_a[29]-K_a_ac*x_a[6]/(K_a_ac+x_a[42]);   /* Sac- */
  f_a[30] = x_a[30]-K_a_co2*x_a[9]/(K_a_co2+x_a[42]); /* SHCO3- */
  f_a[31] = x_a[31]-K_a_IN*x_a[10]/(K_a_IN+x_a[42]);  /* SNH3 */

  f_a[32] = xdot[32]+x_a[32]*q_gas/V_gas-procT8*V_liq/V_gas;  /* Sgas,H2 */
  f_a[33] = xdot[33]+x_a[33]*q_gas/V_gas-procT9*V_liq/V_gas;  /* Sgas,CH4 */
  f_a[34] = xdot[34]+x_a[34]*q_gas/V_gas-procT10*V_liq/V_gas; /* Sgas,CO2 */
        
  f_a[35] = xdot[35]; 											/* Flow is constant*/
  f_a[36] = xdot[36]; 											/* Temp is constant*/

  /* Dummy states*/
  f_a[37] = xdot[37];
  f_a[38] = xdot[38];
  f_a[39] = xdot[39];
  f_a[40] = xdot[40];
  f_a[41] = xdot[41];

  f_a[42] = x_a[24]+(x_a[10]-x_a[31])+x_a[42]-x_a[30]-x_a[29]/64.0-x_a[28]/112.0-x_a[27]/160.0-x_a[26]/208.0-K_w/x_a[42]-x_a[25]; /* SH+ */

  f_a[43] = x_a[43] - (x_a[9] - x_a[30]);				 /* SCO2 */
  f_a[44] = x_a[44] - (x_a[10] - x_a[31]);			 /* SNH4+ */


  f_a[0] >>= f[0];
  f_a[1] >>= f[1];
  f_a[2] >>= f[2];
  f_a[3] >>= f[3];
  f_a[4] >>= f[4];
  f_a[5] >>= f[5];
  f_a[6] >>= f[6];
  f_a[7] >>= f[7];
  f_a[8] >>= f[8];
  f_a[9] >>= f[9];
  f_a[10] >>= f[10];
  f_a[11] >>= f[11];
  f_a[12] >>= f[12];
  f_a[13] >>= f[13];
  f_a[14] >>= f[14];
  f_a[15] >>= f[15];
  f_a[16] >>= f[16];
  f_a[17] >>= f[17];
  f_a[18] >>= f[18];
  f_a[19] >>= f[19];
  f_a[20] >>= f[20];
  f_a[21] >>= f[21];
  f_a[22] >>= f[22];
  f_a[23] >>= f[23];
  f_a[24] >>= f[24];
  f_a[25] >>= f[25];
  f_a[26] >>= f[26];
  f_a[27] >>= f[27];
  f_a[28] >>= f[28];
  f_a[29] >>= f[29];
  f_a[30] >>= f[30];
  f_a[31] >>= f[31];
  f_a[32] >>= f[32];
  f_a[33] >>= f[33];
  f_a[34] >>= f[34];
  f_a[35] >>= f[35];
  f_a[36] >>= f[36];
  f_a[37] >>= f[37];
  f_a[38] >>= f[38];
  f_a[39] >>= f[39];
  f_a[40] >>= f[40];
  f_a[41] >>= f[41];
  f_a[42] >>= f[42];
  f_a[43] >>= f[43];
  f_a[44] >>= f[44];
  trace_off();
  /* End of active section */


  ierr = VecRestoreArrayRead(ctx->influent,&influent);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Udot,&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  'Active' ADOL-C annotated version, marking dependence upon udot.
*/
PetscErrorCode IFunctionActive2(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,AppCtx *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *f;
  const PetscScalar *x,*xdot,*params,*influent; 

  adouble           f_a[45]; /* 'active' double for dependent variables */
  adouble           xdot_a[45]; /* 'active' double for independent variables */

  adouble f_sI_xc, f_xI_xc, f_ch_xc, f_pr_xc, f_li_xc, N_xc, N_I, N_aa, C_xc, C_sI, C_ch, C_pr, C_li, C_xI, C_su, C_aa, f_fa_li, C_fa, f_h2_su, f_bu_su, f_pro_su, f_ac_su, N_bac, C_bu, C_pro, C_ac, C_bac, Y_su, f_h2_aa, f_va_aa, f_bu_aa, f_pro_aa, f_ac_aa, C_va, Y_aa, Y_fa, Y_c4, Y_pro, C_ch4, Y_ac, Y_h2;
  adouble k_dis, k_hyd_ch, k_hyd_pr, k_hyd_li, K_S_IN, k_m_su, K_S_su, pH_UL_aa, pH_LL_aa;
  adouble k_m_aa, K_S_aa, k_m_fa, K_S_fa, K_Ih2_fa, k_m_c4, K_S_c4, K_Ih2_c4, k_m_pro, K_S_pro;
  adouble K_Ih2_pro, k_m_ac, K_S_ac, K_I_nh3, pH_UL_ac, pH_LL_ac, k_m_h2, K_S_h2, pH_UL_h2, pH_LL_h2;
  adouble k_dec_Xsu, k_dec_Xaa, k_dec_Xfa, k_dec_Xc4, k_dec_Xpro, k_dec_Xac, k_dec_Xh2;
  adouble R, T_base, T_op;
  adouble K_H_h2_base, factor; 
  adouble P_atm, p_gas_h2o, P_gas, k_P, kLa, K_H_co2, K_H_ch4, K_H_h2;
  adouble V_liq, V_gas, t_resx, V_frac;
  adouble eps, pH_op, S_H_ion;
  adouble proc1, proc2, proc3, proc4, proc5, proc6, proc7, proc8, proc9, proc10, proc11, proc12, proc13, proc14, proc15, proc16, proc17, proc18, proc19, procT8, procT9, procT10;
  adouble I_pH_aa, I_pH_ac, I_pH_h2, I_IN_lim, I_h2_fa, I_h2_c4, I_h2_pro, I_nh3;
  adouble reac1, reac2, reac3, reac4, reac5, reac6, reac7, reac8, reac9, reac10, reac11, reac12, reac13, reac14, reac15, reac16, reac17, reac18, reac19, reac20, reac21, reac22, reac23, reac24, stoich1, stoich2, stoich3, stoich4, stoich5, stoich6, stoich7, stoich8, stoich9, stoich10, stoich11, stoich12, stoich13;
  adouble inhib[6]; /*declare arrays */
  adouble p_gas_h2, p_gas_ch4, p_gas_co2, q_gas;
  adouble pHLim_aa, pHLim_ac, pHLim_h2, n_aa, n_ac, n_h2;
  // PetscInt i;
  adouble K_w, pK_w_base, K_a_va, pK_a_va_base, K_a_bu, pK_a_bu_base, K_a_pro, pK_a_pro_base, K_a_ac, pK_a_ac_base, K_a_co2, pK_a_co2_base, K_a_IN, pK_a_IN_base;

  eps = 0.0;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(ctx->params,&params);CHKERRQ(ierr);
  f_sI_xc = params[0];
  f_xI_xc = params[1];
  f_ch_xc = params[2];
  f_pr_xc = params[3];
  f_li_xc = params[4];
  N_xc = params[5];
  N_I = params[6];
  N_aa = params[7];
  C_xc = params[8];
  C_sI = params[9];
  C_ch = params[10];
  C_pr = params[11];
  C_li = params[12];
  C_xI = params[13];
  C_su = params[14];
  C_aa = params[15];
  f_fa_li = params[16];
  C_fa = params[17];
  f_h2_su = params[18];
  f_bu_su = params[19];
  f_pro_su = params[20];
  f_ac_su = params[21];
  N_bac = params[22];
  C_bu = params[23];
  C_pro = params[24];
  C_ac = params[25];
  C_bac = params[26];
  Y_su = params[27];
  f_h2_aa = params[28];
  f_va_aa = params[29];
  f_bu_aa = params[30];
  f_pro_aa = params[31];
  f_ac_aa = params[32];
  C_va = params[33];
  Y_aa = params[34];
  Y_fa = params[35];
  Y_c4 = params[36];
  Y_pro = params[37];
  C_ch4 = params[38];
  Y_ac = params[39];
  Y_h2 = params[40];
  k_dis = params[41];
  k_hyd_ch = params[42];
  k_hyd_pr = params[43];
  k_hyd_li = params[44];
  K_S_IN = params[45];
  k_m_su = params[46];
  K_S_su = params[47];
  pH_UL_aa = params[48];
  pH_LL_aa = params[49];
  k_m_aa = params[50];
  K_S_aa = params[51];
  k_m_fa = params[52];
  K_S_fa = params[53];
  K_Ih2_fa = params[54];
  k_m_c4 = params[55];
  K_S_c4 = params[56];
  K_Ih2_c4 = params[57];
  k_m_pro = params[58];
  K_S_pro = params[59];
  K_Ih2_pro = params[60];
  k_m_ac = params[61];
  K_S_ac = params[62];
  K_I_nh3 = params[63];
  pH_UL_ac = params[64];
  pH_LL_ac = params[65];
  k_m_h2 = params[66];
  K_S_h2 = params[67];
  pH_UL_h2 = params[68];
  pH_LL_h2 = params[69];
  k_dec_Xsu = params[70];
  k_dec_Xaa = params[71];
  k_dec_Xfa = params[72];
  k_dec_Xc4 = params[73];
  k_dec_Xpro = params[74];
  k_dec_Xac = params[75];
  k_dec_Xh2 = params[76];
  R = params[77];
  T_base = params[78];
  T_op = params[79];
  pK_w_base = params[80];
  pK_a_va_base = params[81];
  pK_a_bu_base = params[82];
  pK_a_pro_base = params[83];
  pK_a_ac_base = params[84];
  pK_a_co2_base = params[85];
  pK_a_IN_base = params[86];

  P_atm = params[93];
  kLa = params[94];
  K_H_h2_base = params[98];
  k_P = params[99];

  ierr = VecRestoreArrayRead(ctx->params,&params);CHKERRQ(ierr);

  V_liq = ctx->V[0];
  V_gas = ctx->V[1];
  t_resx = ctx->t_resx;


  /* Set the residual */
  ierr = VecGetArrayRead(ctx->influent,&influent);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  /* Start of active section */
  trace_on(2);
  xdot_a[0] <<= xdot[0];
  xdot_a[1] <<= xdot[1];
  xdot_a[2] <<= xdot[2];
  xdot_a[3] <<= xdot[3];
  xdot_a[4] <<= xdot[4];
  xdot_a[5] <<= xdot[5];
  xdot_a[6] <<= xdot[6];
  xdot_a[7] <<= xdot[7];
  xdot_a[8] <<= xdot[8];
  xdot_a[9] <<= xdot[9];
  xdot_a[10] <<= xdot[10];
  xdot_a[11] <<= xdot[11];
  xdot_a[12] <<= xdot[12];
  xdot_a[13] <<= xdot[13];
  xdot_a[14] <<= xdot[14];
  xdot_a[15] <<= xdot[15];
  xdot_a[16] <<= xdot[16];
  xdot_a[17] <<= xdot[17];
  xdot_a[18] <<= xdot[18];
  xdot_a[19] <<= xdot[19];
  xdot_a[20] <<= xdot[20];
  xdot_a[21] <<= xdot[21];
  xdot_a[22] <<= xdot[22];
  xdot_a[23] <<= xdot[23];
  xdot_a[24] <<= xdot[24];
  xdot_a[25] <<= xdot[25];
  xdot_a[26] <<= xdot[26];
  xdot_a[27] <<= xdot[27];
  xdot_a[28] <<= xdot[28];
  xdot_a[29] <<= xdot[29];
  xdot_a[30] <<= xdot[30];
  xdot_a[31] <<= xdot[31];
  xdot_a[32] <<= xdot[32];
  xdot_a[33] <<= xdot[33];
  xdot_a[34] <<= xdot[34];
  xdot_a[35] <<= xdot[35];
  xdot_a[36] <<= xdot[36];
  xdot_a[37] <<= xdot[37];
  xdot_a[38] <<= xdot[38];
  xdot_a[39] <<= xdot[39];
  xdot_a[40] <<= xdot[40];
  xdot_a[41] <<= xdot[41];
  xdot_a[42] <<= xdot[42];
  xdot_a[43] <<= xdot[43];
  xdot_a[44] <<= xdot[44];
  /* Mark independence */



  factor = (1.0/T_base - 1.0/T_op)/(100.0*R);
  K_a_va = pow(10,-pK_a_va_base);
  K_a_bu = pow(10,-pK_a_bu_base);
  K_a_pro = pow(10,-pK_a_pro_base);
  K_w = pow(10,-pK_w_base)*exp(55700.0*factor); /* T adjustment for K_w */
  K_a_ac = pow(10,-pK_a_ac_base)*exp(-4600.0*factor);  /* T adjustment */
  K_a_co2 = pow(10,-pK_a_co2_base)*exp(7600.0*factor); /* T adjustment for K_a_co2 */
  K_a_IN = pow(10,-pK_a_IN_base)*exp(51800.0*factor);  /* T adjustment for K_a_IN */


  K_H_h2 = 1.0/pow(10,(-187.04/T_op+5.473))*55.6/1.01325;     /* atm*L/mol; modified van't Hoff-Arrhenius relationship; MetCalf & Eddy, p68; conversions 55.6 mole H2O/L; 1 atm = 1.01325 */
  K_H_ch4 = 1.0/pow(10,(-675.74/T_op+6.880))*55.6/1.01325; /* atm*L/mol; modified van't Hoff-Arrhenius relationship; MetCalf & Eddy, p68 */
  K_H_co2 = 1.0/pow(10,(-1012.40/T_op+6.606))*55.6/1.01325; /* atm*L/mol; modified van't Hoff-Arrhenius relationship; MetCalf & Eddy, p68 */

  p_gas_h2o = pow(10, (5.20389-1733.926/(T_op-39.485)));  /* Antoine equation for water pressure for temp range 31C-60C */ 

  p_gas_h2 = x[32]*R*T_op/16.0;
  p_gas_ch4 = x[33]*R*T_op/64.0;
  p_gas_co2 = x[34]*R*T_op;
  P_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o;

  S_H_ion = x[42];
  pH_op = -log10(x[42]);   /* pH */


  /* Inhibition */
  /* Hill function on SH+ used within BSM2, ADM1 Workshop, Copenhagen 2005 */
  /*Hill function is replaced by other inhibition terms, assign value to the parameters in Hill function to avoid error*/
//  pHLim_aa = pow(10,(-(pH_UL_aa + pH_LL_aa)/2.0));
//  pHLim_ac = pow(10,(-(pH_UL_ac + pH_LL_ac)/2.0));
//  pHLim_h2 = pow(10,(-(pH_UL_h2 + pH_LL_h2)/2.0));
//  n_aa=3.0/(pH_UL_aa-pH_LL_aa);
//  n_ac=3.0/(pH_UL_ac-pH_LL_ac);
//  n_h2=3.0/(pH_UL_h2-pH_LL_h2);
//  I_pH_aa = pow(pHLim_aa,n_aa)/(pow(S_H_ion,n_aa)+pow(pHLim_aa,n_aa));
//  I_pH_ac = pow(pHLim_ac,n_ac)/(pow(S_H_ion,n_ac)+pow(pHLim_ac,n_ac));
//    I_pH_h2 = pow(pHLim_h2,n_h2)/(pow(S_H_ion,n_h2)+pow(pHLim_h2,n_h2));


  pHLim_aa = 1;
  pHLim_ac = 1;
  pHLim_h2 = 1;
  n_aa=1;
  n_ac=1;
  n_h2=1;
  I_pH_aa = 1;
  I_pH_ac = 1;

  /*New inhibition terms used*/
  I_pH_h2 = (pH_UL_h2) / ( pH_UL_h2 + x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))*60/(64*1000) );


  I_IN_lim = 1.0/(1.0+K_S_IN/x[10]);
  I_h2_fa = 1.0/(1.0+x[7]/K_Ih2_fa);//
  I_h2_c4 = 1.0/(1.0+x[7]/K_Ih2_c4);//
  I_h2_pro = 1.0/(1.0+x[7]/K_Ih2_pro);//
  I_nh3 = 1.0/(1.0+x[31]/K_I_nh3);//

  inhib[0] = I_IN_lim;
//    I_pH_aa*I_IN_lim;
  inhib[1] = inhib[0]*I_h2_fa;
  inhib[2] = inhib[0]*I_h2_c4;
  inhib[3] = inhib[0]*I_h2_pro;
  inhib[4] = I_IN_lim*I_nh3;
//    I_pH_ac*I_IN_lim*I_nh3;
  inhib[5] = I_pH_h2*I_IN_lim;

  /* Process Rates*/  
  proc1 = k_dis*x[12]*exp(3737*(T_op-T_base)/T_op/T_base);
  proc2 = k_hyd_ch*x[13]*exp(3737*(T_op-T_base)/T_op/T_base);
  proc3 = k_hyd_pr*x[14]*exp(3737*(T_op-T_base)/T_op/T_base);
  proc4 = k_hyd_li*x[15]*exp(3737*(T_op-T_base)/T_op/T_base);
    
  proc5 = k_m_su*x[0]/(K_S_su+x[0]*(1 + (x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))) / pH_UL_aa))*x[16]*inhib[0]*exp(3101*(T_op-T_base)/T_op/T_base);
  proc6 = k_m_aa*x[1]/(K_S_aa+x[1]*(1 + (x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))) / pH_UL_aa))*x[17]*inhib[0]*exp(3101*(T_op-T_base)/T_op/T_base);
  proc7 = k_m_fa*x[2]*2370*x[18]/(K_S_fa*2370*x[18]+x[2]*(1 + (x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))) / pH_UL_aa)*2370*x[18]+x[2]*x[2]*x[2])*x[18]*inhib[1]*exp(3101*(T_op-T_base)/T_op/T_base);
  proc8 = k_m_c4*x[3]/(K_S_c4+x[3]*(1 + (x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))) / pH_UL_aa))*x[19]*x[3]/(x[3]+x[4]+eps)*inhib[2]*exp(3101*(T_op-T_base)/T_op/T_base);
  proc9 = k_m_c4*x[4]/(K_S_c4+x[4]*(1 + (x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))) / pH_UL_aa))*x[19]*x[4]/(x[3]+x[4]+eps)*inhib[2]*exp(3101*(T_op-T_base)/T_op/T_base);
  proc10 = k_m_pro*x[5]/(K_S_pro+x[5]*(1 + (x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))) / pH_UL_aa))*x[20]*inhib[3]*exp(6706*(T_op-T_base)/T_op/T_base);
    
    //    Proc5-10 used I_ph_aa; equation from Zhu et al (2018). (Equation 11) factor of (1 + [product]/Km) Km is suspended constant
    //    *(1 + (x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))) / 0.0947) HAc
    
  proc11 = k_m_ac*(x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op)))))/(K_S_ac+x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op))))+pow(x[6]*(1-(1.0/(1.0 + pow(10, pK_a_ac_base - pH_op)))),2)/pH_UL_ac)*x[18]*(2370*x[21]/(2370*x[18]+x[2]*x[2]))*inhib[4]*exp(8184*(T_op-T_base)/T_op/T_base);
//    k_m_ac*x[6]/(K_S_ac+x[6])*x[21]*inhib[4];
  proc12 = k_m_h2*x[7]/(K_S_h2+x[7])*x[22]*(2370*x[18]/(2370*x[18]+x[2]*x[2]))*inhib[5]*exp(4803*(T_op-T_base)/T_op/T_base);//
  proc13 = k_dec_Xsu*x[16];
  proc14 = k_dec_Xaa*x[17];
  proc15 = k_dec_Xfa*x[18];
  proc16 = k_dec_Xc4*x[19];
  proc17 = k_dec_Xpro*x[20];
  proc18 = k_dec_Xac*x[21];
  proc19 = k_dec_Xh2*x[22];


  procT8 = kLa*(x[7]-16.0*K_H_h2*p_gas_h2)*pow(1.012,(T_op-T_base)); //
  procT9 = kLa*(x[8]-64.0*K_H_ch4*p_gas_ch4)*pow(1.012,(T_op-T_base)); //
  procT10 = kLa*((x[43])-K_H_co2*p_gas_co2)*pow(1.012,(T_op-T_base));//

  /* Stoichiometry for C and N balance */ 
  stoich1 = -C_xc+f_sI_xc*C_sI+f_ch_xc*C_ch+f_pr_xc*C_pr+f_li_xc*C_li+f_xI_xc*C_xI;
  stoich2 = -C_ch+C_su;
  stoich3 = -C_pr+C_aa;
  stoich4 = -C_li+(1.0-f_fa_li)*C_su+f_fa_li*C_fa;
  stoich5 = -C_su+(1.0-Y_su)*(f_bu_su*C_bu+f_pro_su*C_pro+f_ac_su*C_ac)+Y_su*C_bac;
  stoich6 = -C_aa+(1.0-Y_aa)*(f_va_aa*C_va+f_bu_aa*C_bu+f_pro_aa*C_pro+f_ac_aa*C_ac)+Y_aa*C_bac;
  stoich7 = -C_fa+(1.0-Y_fa)*0.7*C_ac+Y_fa*C_bac;
  stoich8 = -C_va+(1.0-Y_c4)*0.54*C_pro+(1.0-Y_c4)*0.31*C_ac+Y_c4*C_bac;
  stoich9 = -C_bu+(1.0-Y_c4)*0.8*C_ac+Y_c4*C_bac;
  stoich10 = -C_pro+(1.0-Y_pro)*0.57*C_ac+Y_pro*C_bac;
  stoich11 = -C_ac+(1.0-Y_ac)*C_ch4+Y_ac*C_bac;
  stoich12 = (1.0-Y_h2)*C_ch4+Y_h2*C_bac;
  stoich13 = -C_bac+C_xc;

  /*Overall Reaction Rates for State Variables*/
  reac1 = proc2+(1.0-f_fa_li)*proc4-proc5;
  reac2 = proc3-proc6;
  reac3 = f_fa_li*proc4-proc7;
  reac4 = (1.0-Y_aa)*f_va_aa*proc6-proc8;
  reac5 = (1.0-Y_su)*f_bu_su*proc5+(1.0-Y_aa)*f_bu_aa*proc6-proc9;
  reac6 = (1.0-Y_su)*f_pro_su*proc5+(1.0-Y_aa)*f_pro_aa*proc6+(1.0-Y_c4)*0.54*proc8-proc10;
  reac7 = (1.0-Y_su)*f_ac_su*proc5+(1.0-Y_aa)*f_ac_aa*proc6+(1.0-Y_fa)*0.7*proc7+(1.0-Y_c4)*0.31*proc8+(1.0-Y_c4)*0.8*proc9+(1.0-Y_pro)*0.57*proc10-proc11;
  /*reac 8 is not used because hydrogen is solved algebraically  reac8 = (1.0-Y_su)*f_h2_su*proc5+(1.0-Y_aa)*f_h2_aa*proc6+(1.0-Y_fa)*0.3*proc7+(1.0-Y_c4)*0.15*proc8+(1.0-Y_c4)*0.2*proc9+(1.0-Y_pro)*0.43*proc10-proc12-procT8;*/
  reac8 = (1.0-Y_su)*f_h2_su*proc5+(1.0-Y_aa)*f_h2_aa*proc6+(1.0-Y_fa)*0.3*proc7+(1.0-Y_c4)*0.15*proc8+(1.0-Y_c4)*0.2*proc9+(1.0-Y_pro)*0.43*proc10-proc12-procT8;
  reac9 = (1.0-Y_ac)*proc11+(1.0-Y_h2)*proc12-procT9;
  reac10 = -stoich1*proc1-stoich2*proc2-stoich3*proc3-stoich4*proc4-stoich5*proc5-stoich6*proc6-stoich7*proc7-stoich8*proc8-stoich9*proc9-stoich10*proc10-stoich11*proc11-stoich12*proc12-stoich13*proc13-stoich13*proc14-stoich13*proc15-stoich13*proc16-stoich13*proc17-stoich13*proc18-stoich13*proc19-procT10;
  reac11 = (N_xc-f_xI_xc*N_I-f_sI_xc*N_I-f_pr_xc*N_aa)*proc1-Y_su*N_bac*proc5+(N_aa-Y_aa*N_bac)*proc6-Y_fa*N_bac*proc7-Y_c4*N_bac*proc8-Y_c4*N_bac*proc9-Y_pro*N_bac*proc10-Y_ac*N_bac*proc11-Y_h2*N_bac*proc12+(N_bac-N_xc)*(proc13+proc14+proc15+proc16+proc17+proc18+proc19);
  reac12 = f_sI_xc*proc1;
  reac13 = -proc1+proc13+proc14+proc15+proc16+proc17+proc18+proc19;
  reac14 = f_ch_xc*proc1-proc2;
  reac15 = f_pr_xc*proc1-proc3;
  reac16 = f_li_xc*proc1-proc4;
  reac17 = Y_su*proc5-proc13;
  reac18 = Y_aa*proc6-proc14;
  reac19 = Y_fa*proc7-proc15;
  reac20 = Y_c4*proc8+Y_c4*proc9-proc16;
  reac21 = Y_pro*proc10-proc17;
  reac22 = Y_ac*proc11-proc18;
  reac23 = Y_h2*proc12-proc19;
  reac24 = f_xI_xc*proc1;

  q_gas = k_P*(P_gas-P_atm);
  /*if (q_gas < 0)
      q_gas = 0.0;*/

  V_frac=1.0/V_liq;

  /* State Variables: q/V*(Sin - S) + reaction(s) */  
  //f_a[0] = xdot_a[0]-1.0/V_liq*(influent[26]*(influent[0]-x[0]))-reac1; 		/* Ssu */
  //f_a[1] = xdot_a[1]-1.0/V_liq*(influent[26]*(influent[1]-x[1]))-reac2; 		/* Saa */
  //f_a[2] = xdot_a[2]-1.0/V_liq*(influent[26]*(influent[2]-x[2]))-reac3;		/* Sfa */
  //f_a[3] = xdot_a[3]-1.0/V_liq*(influent[26]*(influent[3]-x[3]))-reac4;  		/* Sva */
  //f_a[4] = xdot_a[4]-1.0/V_liq*(influent[26]*(influent[4]-x[4]))-reac5;  		/* Sbu */
  //f_a[5] = xdot_a[5]-1.0/V_liq*(influent[26]*(influent[5]-x[5]))-reac6;  		/* Spro */
  //f_a[6] = xdot_a[6]-1.0/V_liq*(influent[26]*(influent[6]-x[6]))-reac7;  		/* Sac */
  //f_a[7] = 1.0/V_liq*(influent[26]*(influent[7]-x[7]))-reac8;			/* Sh2 */
  //f_a[8] = xdot_a[8]-1.0/V_liq*(influent[26]*(influent[8]-x[8]))-reac9;     	/* Sch4 */
  //f_a[9] = xdot_a[9]-1.0/V_liq*(influent[26]*(influent[9]-x[9]))-reac10;   	/* SIC */
  //f_a[10] = xdot_a[10]-1.0/V_liq*(influent[26]*(influent[10]-x[10]))-reac11; 	/* SIN */
  //f_a[11] = xdot_a[11]-1.0/V_liq*(influent[26]*(influent[11]-x[11]))-reac12;  	/* SI */
  //f_a[12] = xdot_a[12]-1.0/V_liq*(influent[26]*(influent[12]-x[12]))-reac13; 	/* Xxc */
  //f_a[13] = xdot_a[13]-1.0/V_liq*(influent[26]*(influent[13]-x[13]))-reac14; 	/* Xch */
  //f_a[14] = xdot_a[14]-1.0/V_liq*(influent[26]*(influent[14]-x[14]))-reac15; 	/* Xpr */
  //f_a[15] = xdot_a[15]-1.0/V_liq*(influent[26]*(influent[15]-x[15]))-reac16; 	/* Xli */
  //f_a[16] = xdot_a[16]-1.0/V_liq*(influent[26]*(influent[16]-x[16]))-reac17; 	/* Xsu */
  //f_a[17] = xdot_a[17]-1.0/V_liq*(influent[26]*(influent[17]-x[17]))-reac18; 	/* Xaa */
  //f_a[18] = xdot_a[18]-1.0/V_liq*(influent[26]*(influent[18]-x[18]))-reac19; 	/* Xfa */
  //f_a[19] = xdot_a[19]-1.0/V_liq*(influent[26]*(influent[19]-x[19]))-reac20; 	/* Xc4 */
  //f_a[20] = xdot_a[20]-1.0/V_liq*(influent[26]*(influent[20]-x[20]))-reac21; 	/* Xpro */
  //f_a[21] = xdot_a[21]-1.0/V_liq*(influent[26]*(influent[21]-x[21]))-reac22;	/* Xac */
  //f_a[22] = xdot_a[22]-1.0/V_liq*(influent[26]*(influent[22]-x[22]))-reac23; 	/* Xh2 */
  //f_a[23] = xdot_a[23]-1.0/V_liq*(influent[26]*(influent[23]-x[23]))-reac24; 	/* XI */
  //f_a[24] = xdot_a[24]-1.0/V_liq*(influent[26]*(influent[24]-x[24])); 			/* Scat+ */
  //if (ctx->set_Cat_mass) {
  //  f_a[24] = f_a[24]+ctx->Cat_mass; 			/* Scat+ */
  //}
  //f_a[25] = xdot_a[25]-1.0/V_liq*(influent[26]*(influent[25]-x[25])); 			/* San- */

  /* State Variables: q/V*(Sin - S) + reaction(s) (Elchin added)*/  
  f_a[0] = xdot_a[0]-V_frac*(influent[26]*(influent[0]-x[0]))-reac1; 		/* Ssu */
  f_a[1] = xdot_a[1]-V_frac*(influent[26]*(influent[1]-x[1]))-reac2; 		/* Saa */
  f_a[2] = xdot_a[2]-V_frac*(influent[26]*(influent[2]-x[2]))-reac3;		/* Sfa */
  f_a[3] = xdot_a[3]-V_frac*(influent[26]*(influent[3]-x[3]))-reac4;  		/* Sva */
  f_a[4] = xdot_a[4]-V_frac*(influent[26]*(influent[4]-x[4]))-reac5;  		/* Sbu */
  f_a[5] = xdot_a[5]-V_frac*(influent[26]*(influent[5]-x[5]))-reac6;  		/* Spro */
  f_a[6] = xdot_a[6]-V_frac*(influent[26]*(influent[6]-x[6]))-reac7;  		/* Sac */
  f_a[7] = V_frac*(influent[26]*(influent[7]-x[7]))-reac8;			/* Sh2 */
  f_a[8] = xdot_a[8]-V_frac*(influent[26]*(influent[8]-x[8]))-reac9;     	/* Sch4 */
  f_a[9] = xdot_a[9]-V_frac*(influent[26]*(influent[9]-x[9]))-reac10;   	/* SIC */
  f_a[10] = xdot_a[10]-V_frac*(influent[26]*(influent[10]-x[10]))-reac11; 	/* SIN */
  f_a[11] = xdot_a[11]-V_frac*(influent[26]*(influent[11]-x[11]))-reac12;  	/* SI */
  f_a[12] = xdot_a[12]-( V_frac*influent[26]*influent[12]-x[12]/(V_liq/influent[26] + t_resx))-reac13; 	/* Xxc */
  f_a[13] = xdot_a[13]-( V_frac*influent[26]*influent[13]-x[13]/(V_liq/influent[26] + t_resx))-reac14; 	/* Xch */
  f_a[14] = xdot_a[14]-( V_frac*influent[26]*influent[14]-x[14]/(V_liq/influent[26] + t_resx))-reac15; 	/* Xpr */
  f_a[15] = xdot_a[15]-( V_frac*influent[26]*influent[15]-x[15]/(V_liq/influent[26] + t_resx))-reac16; 	/* Xli */
  f_a[16] = xdot_a[16]-( V_frac*influent[26]*influent[16]-x[16]/(V_liq/influent[26] + t_resx))-reac17; 	/* Xsu */
  f_a[17] = xdot_a[17]-( V_frac*influent[26]*influent[17]-x[17]/(V_liq/influent[26] + t_resx))-reac18; 	/* Xaa */
  f_a[18] = xdot_a[18]-( V_frac*influent[26]*influent[18]-x[18]/(V_liq/influent[26] + t_resx))-reac19; 	/* Xfa */
  f_a[19] = xdot_a[19]-( V_frac*influent[26]*influent[19]-x[19]/(V_liq/influent[26] + t_resx))-reac20; 	/* Xc4 */
  f_a[20] = xdot_a[20]-( V_frac*influent[26]*influent[20]-x[20]/(V_liq/influent[26] + t_resx))-reac21; 	/* Xpro */
  f_a[21] = xdot_a[21]-( V_frac*influent[26]*influent[21]-x[21]/(V_liq/influent[26] + t_resx))-reac22;	/* Xac */
  f_a[22] = xdot_a[22]-( V_frac*influent[26]*influent[22]-x[22]/(V_liq/influent[26] + t_resx))-reac23; 	/* Xh2 */
  f_a[23] = xdot_a[23]-( V_frac*influent[26]*influent[23]-x[23]/(V_liq/influent[26] + t_resx))-reac24; 	/* XI */
  f_a[24] = xdot_a[24]-V_frac*(influent[26]*(influent[24]-x[24]));            /* Scat+ */

  if (ctx->set_Cat_mass) {
    f_a[24] = f_a[24]+ctx->Cat_mass; 			/* Scat+ */
  }
  f_a[25] = xdot_a[25]-V_frac*(influent[26]*(influent[25]-x[25])); 			/* San- */

  f_a[26] = x[26]-K_a_va*x[3]/(K_a_va+x[42]);   /* Sva- */
  f_a[27] = x[27]-K_a_bu*x[4]/(K_a_bu+x[42]);   /* Sbu- */
  f_a[28] = x[28]-K_a_pro*x[5]/(K_a_pro+x[42]); /* Spro- */
  f_a[29] = x[29]-K_a_ac*x[6]/(K_a_ac+x[42]);   /* Sac- */
  f_a[30] = x[30]-K_a_co2*x[9]/(K_a_co2+x[42]); /* SHCO3- */
  f_a[31] = x[31]-K_a_IN*x[10]/(K_a_IN+x[42]);  /* SNH3 */

  f_a[32] = xdot_a[32]+x[32]*q_gas/V_gas-procT8*V_liq/V_gas;  /* Sgas,H2 */
  f_a[33] = xdot_a[33]+x[33]*q_gas/V_gas-procT9*V_liq/V_gas;  /* Sgas,CH4 */
  f_a[34] = xdot_a[34]+x[34]*q_gas/V_gas-procT10*V_liq/V_gas; /* Sgas,CO2 */
        
  f_a[35] = xdot_a[35]; 											/* Flow is constant*/
  f_a[36] = xdot_a[36]; 											/* Temp is constant*/

  /* Dummy states*/
  f_a[37] = xdot_a[37];
  f_a[38] = xdot_a[38];
  f_a[39] = xdot_a[39];
  f_a[40] = xdot_a[40];
  f_a[41] = xdot_a[41];

  f_a[42] = x[24]+(x[10]-x[31])+x[42]-x[30]-x[29]/64.0-x[28]/112.0-x[27]/160.0-x[26]/208.0-K_w/x[42]-x[25]; /* SH+ */

  f_a[43] = x[43] - (x[9] - x[30]);				 /* SCO2 */
  f_a[44] = x[44] - (x[10] - x[31]);			 /* SNH4+ */


  f_a[0] >>= f[0];
  f_a[1] >>= f[1];
  f_a[2] >>= f[2];
  f_a[3] >>= f[3];
  f_a[4] >>= f[4];
  f_a[5] >>= f[5];
  f_a[6] >>= f[6];
  f_a[7] >>= f[7];
  f_a[8] >>= f[8];
  f_a[9] >>= f[9];
  f_a[10] >>= f[10];
  f_a[11] >>= f[11];
  f_a[12] >>= f[12];
  f_a[13] >>= f[13];
  f_a[14] >>= f[14];
  f_a[15] >>= f[15];
  f_a[16] >>= f[16];
  f_a[17] >>= f[17];
  f_a[18] >>= f[18];
  f_a[19] >>= f[19];
  f_a[20] >>= f[20];
  f_a[21] >>= f[21];
  f_a[22] >>= f[22];
  f_a[23] >>= f[23];
  f_a[24] >>= f[24];
  f_a[25] >>= f[25];
  f_a[26] >>= f[26];
  f_a[27] >>= f[27];
  f_a[28] >>= f[28];
  f_a[29] >>= f[29];
  f_a[30] >>= f[30];
  f_a[31] >>= f[31];
  f_a[32] >>= f[32];
  f_a[33] >>= f[33];
  f_a[34] >>= f[34];
  f_a[35] >>= f[35];
  f_a[36] >>= f[36];
  f_a[37] >>= f[37];
  f_a[38] >>= f[38];
  f_a[39] >>= f[39];
  f_a[40] >>= f[40];
  f_a[41] >>= f[41];
  f_a[42] >>= f[42];
  f_a[43] >>= f[43];
  f_a[44] >>= f[44];
  trace_off();
  /* End of active section */


  ierr = VecRestoreArrayRead(ctx->influent,&influent);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Udot,&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 Defines the Jacobian of the ODE passed to the ODE solver, using the PETSc-ADOL-C driver for
 implicit TS.
*/
PetscErrorCode IJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat B,AppCtx *ctx)
{
  PetscErrorCode ierr;
  AppCtx         *appctx = (AppCtx*)ctx;
  const PetscScalar    *u;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = PetscAdolcComputeIJacobian(1,2,A,u,a,appctx->adctx);CHKERRQ(ierr);
  // ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode InitialConditions(TS ts,Vec U,AppCtx *ctx)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecCopy(ctx->initialconditions,U);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* ODE integrator */
  Vec            U,Udot,R;      /* solution, derivative, residual */
  Mat            A;             /* Jacobian matrix */
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n = 45;
  PetscInt       num_adm1out = 54;
  PetscInt       num_asm1out = 56;
  PetscInt       num_indicators = 67;
  PetscBool      steady, set_Vliq, set_Vgas, set_t_resx;
  AppCtx         ctx;
  // PetscScalar    *u;
  AdolcCtx       *adctx;
  char           params_file[PETSC_MAX_PATH_LEN]="params.dat";
  char           influent_file[PETSC_MAX_PATH_LEN]="influent.dat";
  char           ic_file[PETSC_MAX_PATH_LEN]="ic.dat";
  PetscReal      Vliq = 3400.0, Vliq_read; // Reactor liquid volume
  PetscReal      Vgas = 300.0, Vgas_read;  // Reactor gas volume
  PetscReal      t_resx = 0, t_resx_read; // SRT adjustment: t_resx = SRT- HRT
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for sequential runs");
  ierr = PetscNew(&adctx);CHKERRQ(ierr);
  adctx->m = n;adctx->n = n;adctx->p = n;
  ctx.adctx = adctx;


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  set_Vliq = PETSC_FALSE;
  set_Vgas = PETSC_FALSE;
  set_t_resx = PETSC_FALSE;
  ierr = PetscOptionsGetReal(NULL,NULL,"-Cat",&ctx.Cat_mass,&ctx.set_Cat_mass);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-Vliq",&Vliq_read,&set_Vliq);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-Vgas",&Vgas_read,&set_Vgas);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-t_resx",&t_resx_read,&set_t_resx);CHKERRQ(ierr);
  
  ierr = PetscOptionsGetString(NULL,NULL,"-params_file",
                                params_file,PETSC_MAX_PATH_LEN,
                                NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-influent_file",
                                influent_file,PETSC_MAX_PATH_LEN,
                                NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-ic_file",
                                ic_file,PETSC_MAX_PATH_LEN,
                                NULL);CHKERRQ(ierr);

  steady = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-steady",&steady,NULL);CHKERRQ(ierr);
  ctx.debug = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-debug",&ctx.debug,NULL);CHKERRQ(ierr);
  
  if (set_Vliq) Vliq = Vliq_read;
  if (set_Vgas) Vgas = Vgas_read;
  if (set_t_resx) t_resx = t_resx_read;
  ierr = PetscPrintf(MPI_COMM_SELF,"Vliq [m3] is: %f\n", Vliq);CHKERRQ(ierr);
  ierr = PetscPrintf(MPI_COMM_SELF,"Vgas [m3] is: %f\n", Vgas);CHKERRQ(ierr);
  ierr = PetscPrintf(MPI_COMM_SELF,"SRT adjestment [d] is: %f\n", t_resx);CHKERRQ(ierr);
  //ierr = PetscPrintf(MPI_COMM_SELF,"HEre I am  is: %f\n", Vgas);CHKERRQ(ierr);
  

  // Read params, influent and ic
  ierr = ReadParams(&ctx,params_file);CHKERRQ(ierr);
  ierr = DigestParToInterfacePar(&ctx);CHKERRQ(ierr);
  ierr = ReadInfluent(&ctx,influent_file);CHKERRQ(ierr);
  ierr = ReadInitialConditions(&ctx,ic_file);CHKERRQ(ierr);

  // Set Volumes
  ctx.V[0] = Vliq;
  ctx.V[1] = Vgas;
  // Set SRT Adjustment
  ctx.t_resx = t_resx;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&U,NULL);CHKERRQ(ierr);

  /* post process vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&ctx.asm1_output);CHKERRQ(ierr); 
  ierr = VecSetSizes(ctx.asm1_output,PETSC_DECIDE,num_asm1out);CHKERRQ(ierr);
  ierr = VecSetFromOptions(ctx.asm1_output);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&ctx.adm1_output);CHKERRQ(ierr); 
  ierr = VecSetSizes(ctx.adm1_output,PETSC_DECIDE,num_adm1out);CHKERRQ(ierr);
  ierr = VecSetFromOptions(ctx.adm1_output);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&ctx.indicator);CHKERRQ(ierr); 
  ierr = VecSetSizes(ctx.indicator,PETSC_DECIDE,num_indicators);CHKERRQ(ierr);
  ierr = VecSetFromOptions(ctx.indicator);CHKERRQ(ierr);

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);

  if (steady) {
    ierr = PetscPrintf(MPI_COMM_SELF,"Running as steady state problem.\n");CHKERRQ(ierr);
    ierr = TSSetType(ts,TSPSEUDO);CHKERRQ(ierr); /* Need a flag that checks if steady or not is to solved */
  } else {
    ierr = PetscPrintf(MPI_COMM_SELF,"Running as transient problem. Use -ts_monitor to see the timestep information.\n");CHKERRQ(ierr);
    ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  }
  // ierr = TSSetEquationType(ts,TS_EQ_DAE_IMPLICIT_INDEX1);CHKERRQ(ierr);

  ierr = TSSetIFunction(ts,NULL,(TSIFunction) IFunctionPassive,&ctx);CHKERRQ(ierr);

  // /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //     Set initial conditions
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = InitialConditions(ts,U,&ctx);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Trace just once for each tape
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDuplicate(U,&Udot);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&R);CHKERRQ(ierr);
  ierr = IFunctionActive1(ts,0.,U,Udot,R,&ctx);CHKERRQ(ierr);
  ierr = IFunctionActive2(ts,0.,U,Udot,R,&ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&R);CHKERRQ(ierr);
  ierr = VecDestroy(&Udot);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set Jacobian
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetIJacobian(ts,A,A,(TSIJacobian)IJacobian,&ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetTimeStep(ts,.001);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,10000);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1e12);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(ts,&ctx);CHKERRQ(ierr);
  ierr = TSSetPostStep(ts,PostProcess);CHKERRQ(ierr);
  ierr = PostProcess(ts);CHKERRQ(ierr); /* print the initial state */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscPrintf(MPI_COMM_SELF,"Solving.\n");CHKERRQ(ierr);
  ierr = TSSolve(ts,U);CHKERRQ(ierr);
  ierr = TSView(ts,PETSC_VIEWER_BINARY_WORLD);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&ctx.initialconditions);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx.params);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx.interface_params);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx.influent);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx.adm1_output);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx.asm1_output);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx.indicator);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFree(adctx);CHKERRQ(ierr);
  ierr = PetscPrintf(MPI_COMM_SELF,"Done!\n");CHKERRQ(ierr);

  ierr = PetscFinalize();
  
  return ierr;
}


/*TEST

   test:
     args: -ts_view
     requires: dlsym define(PETSC_HAVE_DYNAMIC_LIBRARIES)

   test:
     suffix: 2
     args: -ts_monitor_lg_error -ts_monitor_lg_solution  -ts_view
     requires: x
     output_file: output/ex1_1.out
     requires: dlsym define(PETSC_HAVE_DYNAMIC_LIBRARIES)
TEST*/
