# Paralelní programování na GPU (PCG 2021)
# Projekt c. 1 (cuda)
# Login: xducho07


N=4096
#N=25600
#N=28160
#N=30720
#N=33280
#N=35840
#N=38400
#N=40960
#N=43520
#N=46080
#N=48640
#N=51200
#N=53760
#N=56320
#N=58880
#N=61440
N=64000
DT=0.01f
STEPS=500
THREADS_PER_BLOCK=512
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
	#-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_x_final
	#-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_y_final
	#-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_z_final
	#-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_w_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /pos_x
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /pos_y
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /pos_z
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /vel_x
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /vel_x
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /vel_x
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /weight
	#-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_x
	#-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_y
	#-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_z
	#-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_w

check_com:
	./nbody 10 0.01f 10000 ${THREADS_PER_BLOCK} 0 ${RED_THREADS} ${RED_THREADS_PER_BLOCK} $(SAMPLE_INPUT) $(OUTPUT)


profile:
	nvprof --print-gpu-trace \
		--metrics flops_sp \
		--metrics flops_sp_add \
		--metrics shared_load_transactions \
		--metrics flop_sp_efficiency \
		--metrics sm_efficiency \
		--metrics alu_fu_utilization \
		--metrics gld_transactions \
		--metrics l1_cache_global_hit_rate \
		--metrics stall_memory_dependency  \
		--metrics gld_throughput  \
		--metrics l2_read_throughput  \
		--metrics gld_troughput  \
		./nbody ${N} ${DT} 1 ${THREADS_PER_BLOCK} 0  ${RED_THREADS} ${RED_THREADS_PER_BLOCK} $(SAMPLE_INPUT) $(OUTPUT) 
