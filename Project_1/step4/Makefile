# Paralelní programování na GPU (PCG 2021)
# Projekt c. 1 (cuda)
# Login: xducho07


N=2048
N=4096
N=8192
N=16384
N=32768
N=65536
N=131072
N=262144
N=524288
N=1048576
N=4096
DT=0.01f
STEPS=50
THREADS_PER_BLOCK=1024
RED_THREADS=4096
RED_THREADS_PER_BLOCK=128
WRITE_INTESITY=20

SAMPLE_INPUT=../sampledata/sampleInput$(N).h5
#SAMPLE_INPUT=../sampledata/sampleInput.h5
SAMPLE_OUTPUT=../sampledata/sampleOutput.h5
OUTPUT=step0Output.h5

INCLUDE=../commons
LIBS=-lhdf5

FLAGS=  

.PHONY: all clean run profile

all: nbody

nbody: nbody.cu main.cu nbody.h
	nvcc ${FLAGS} -I${INCLUDE} nbody.cu main.cu ../commons/h5Helper.cpp ${LIBS} -o nbody

clean:
	rm -f *.o nbody

run:
	./nbody ${N} ${DT} ${STEPS} ${THREADS_PER_BLOCK} ${WRITE_INTESITY} ${RED_THREADS} ${RED_THREADS_PER_BLOCK} $(SAMPLE_INPUT) $(OUTPUT)

check_output:
	./nbody 4096 0.01f 500 ${THREADS_PER_BLOCK} 20 ${RED_THREADS} ${RED_THREADS_PER_BLOCK} $(SAMPLE_INPUT) $(OUTPUT)
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /pos_x_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /pos_y_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /pos_z_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /vel_x_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /vel_x_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /vel_x_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /weight_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_x_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_y_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_z_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_w_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /pos_x
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /pos_y
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /pos_z
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /vel_x
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /vel_x
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /vel_x
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /weight
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_x
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_y
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_z
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_w

check_com:
	./nbody 10 0.01f 10000 ${THREADS_PER_BLOCK} 0 ${RED_THREADS} ${RED_THREADS_PER_BLOCK} $(SAMPLE_INPUT) $(OUTPUT)


profile:
	nvprof --print-gpu-trace \
		--devices 0 \
		--metrics flops_sp \
		--metrics flop_sp_efficiency \
		--metrics sm_efficiency \
		--metrics alu_fu_utilization \
		--metrics gld_transactions \
		--metrics l1_cache_global_hit_rate \
		--metrics shared_load_transactions \
		--metrics stall_memory_dependency  \
		--metrics gld_throughput  \
		--metrics l2_read_throughput  \
		./nbody ${N} ${DT} 1 ${THREADS_PER_BLOCK} 0  ${RED_THREADS} ${RED_THREADS_PER_BLOCK} $(SAMPLE_INPUT) $(OUTPUT) 
