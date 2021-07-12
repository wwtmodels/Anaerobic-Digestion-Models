#requiresdefine   'PETSC_HAVE_ATTRIBUTEALIGNED'

ALL:adm1f
CFLAGS	         = 
FFLAGS	         = 
CPPFLAGS         = 
FPPFLAGS         =
EXAMPLESC        = adm1f_srt.cxx
LIB             = ${PETSC_TS_LIB}


OBJECTS_PF = adm1f_srt.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

adm1f: adm1f_srt.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

#--------------------------------------------------------------------------
run_adm1f:
	-@${MPIEXEC} -n 1 ./adm1f -ts_monitor

clean::
	rm -f adm1f
