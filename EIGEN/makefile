gcc_options = -std=c++17 -Wall --pedantic-errors -DMKL_ILP64  -I"${MKLROOT}/include" -g
l_b = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

program : EIGENTest.o EIGEN.o
	icpc -o $@ $^ $(l_b)

EIGENTest.o : EIGENTest.cpp
	icpc -c $< $(l_b)

EIGEN.o : EIGEN.cpp
	icpc -c $< $(l_b)

Jset.o : Jset.cpp
	icpc -c $< $(l_b)
	
run : program
	./program

clean:
	rm -f ./program

.PHONY : run clean
