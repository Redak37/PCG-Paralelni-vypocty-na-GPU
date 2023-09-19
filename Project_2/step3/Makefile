#/**
# * @file      Makefile
# *
# * @author    Radek Duchon
# *            Faculty of Information Technology
# *            Brno University of Technology
# *            xducho07@stud.fit.vutbr.cz
# *
# * @brief     PCG Assignment 2
# *            N-Body simulation in ACC - Makefile
# *
# * @version   2021
# *
# * @date      11 November  2020, 11:22 (created) \n
# * @date      11 November  2020, 11:37 (revised) \n
# *
# */

N=1024
N=2048
N=3072
N=4096
N=5120
N=6144
N=7168
N=8192
N=9216
N=10240
N=11624
N=12288
N=13312
N=14336
N=15360
N=16384
N=17408
N=18432
N=19456
N=20480
N=21504
N=22528
N=23552
N=24756
N=25600
N=26624
N=27648
N=28672
N=29696
N=30720

#N=32768
#N=65536
#N=131072

#N=64000
DT=0.01f
STEPS=500
WRITE_INTESITY=20

INPUT=../sampledata/sampleInput$(N).h5
SAMPLE_INPUT=../sampledata/sampleInput$(N).h5
OUTPUT=step1Output.h5
SAMPLE_OUTPUT=../sampledata/sampleOutput.h5

INCLUDE=../commons
LIBS=-lhdf5 -lm

FLAGS=  

.PHONY: all clean run profile


CC=nvc++
CFLAGS=-march=native --c++14

# Acceleration for Barbora GPUs
ACC=-acc=gpu -gpu=cc70,fastmath,pinned

# Acceleration for Barbora GPUs
#ACC=-acc=multicore

OPT=-O3
LOG=-Minfo=accel

# OpenACC acceleration types
#[no]acc[=gpu|host|multicore|[no]autopar|[no]routineseq|legacy|strict|verystrict|sync|[no]wait]
#                    Enable OpenACC directives
#    gpu             OpenACC directives are compiled for GPU execution only; please refer to -gpu for target specific options
#    host            Compile for serial execution on the host CPU
#    multicore       Compile for parallel execution on the host CPU
#    [no]autopar     Enable (default) or disable loop autoparallelization within acc parallel
#    [no]routineseq  Compile every routine for the device
#    legacy          Suppress warnings about deprecated PGI accelerator directives
#    strict          Issue warnings for non-OpenACC accelerator directives
#    verystrict      Fail with an error for any non-OpenACC accelerator directive
#    sync            Ignore async clauses
#    [no]wait        Wait for each device kernel to finish
#-cuda[=charstring|madconst]
#                    Enable CUDA; please refer to -gpu for target specific options
#    charstring      Enable limited support for character strings in GPU kernels
#    madconst        Put Module Array Descriptors in CUDA Constant Memory

#GPU architecture
#-gpu=cc35|cc50|cc60|cc62|cc70|cc72|cc75|cc80|ccall|cudaX.Y|fastmath|[no]flushz|[no]fma|keep|[no]lineinfo|llc|zeroinit|[no]autocollapse|deepcopy|loadcache:{L1|L2}|maxregcount:<n>|pinned|[no]rdc|safecache|[no]unroll|[no]managed|beta|autocompare|redundant
#                    Select specific options for GPU code generation
#    cc35            Compile for compute capability 3.5
#    cc50            Compile for compute capability 5.0
#    cc60            Compile for compute capability 6.0
#    cc62            Compile for compute capability 6.2
#    cc70            Compile for compute capability 7.0
#    cc72            Compile for compute capability 7.2
#    cc75            Compile for compute capability 7.5
#    cc80            Compile for compute capability 8.0
#    ccall           Compile for all supported compute capabilities
#    cudaX.Y         Use CUDA X.Y Toolkit compatibility, where installed
#    fastmath        Use fast math library
#    [no]flushz      Enable flush-to-zero mode on the GPU
#    [no]fma         Generate fused mul-add instructions (default at -O3)
#    keep            Keep kernel files
#    [no]lineinfo    Generate GPU line information
#    zeroinit        Initialize allocated device memory with zero
#    [no]autocollapse
#                    Automatically collapse tightly nested parallel loops
#    deepcopy        Enable Full Deepcopy support in OpenACC Fortran
#    loadcache       Choose what hardware level cache to use for global memory loads
#     L1             Use L1 cache
#     L2             Use L2 cache
#    maxregcount:<n> Set maximum number of registers to use on the GPU
#    pinned          Use CUDA Pinned Memory
#    [no]rdc         Generate relocatable device code
#    safecache       Allows variable-sized array sections in cache directives and assumes they fit into CUDA shared memory
#    [no]unroll      Enable automatic inner loop unrolling (default at -O3)
#    [no]managed     Use CUDA Managed Memory
#    beta            Enable beta code generation features
#    autocompare     Automatically compare CPU/GPU results: implies redundant
#    redundant       Redundant CPU/GPU execution

all: nbody

nbody: nbody.cpp main.cpp nbody.h
	${CC} ${CFLAGS} ${OPT} ${ACC} ${LOG} -I${INCLUDE} nbody.cpp main.cpp ../commons/h5Helper.cpp ${LIBS} -o nbody

clean:
	rm -f *.o nbody

run:
	./nbody ${N} ${DT} ${STEPS} ${WRITE_INTESITY} $(INPUT) $(OUTPUT)

profile:
	nvprof --print-gpu-trace\
	     --devices 0 \
	     --metrics flops_sp \
	     --metrics flops_sp_add \
	     --metrics flops_sp_mul \
	     --metrics flops_sp_fma \
	     --metrics flops_sp_special \
	     --metrics flop_sp_efficiency \
	     --metrics sm_efficiency \
	     --metrics alu_fu_utilization \
	     --metrics gld_transactions \
	     --metrics gld_efficiency \
	     --metrics achieved_occupancy \
	     --metrics shared_efficiency \
	     --metrics gld_requested_throughput	\
	     --metrics gst_requested_throughput \
	     --metrics l2_read_throughput \
	     --metrics l2_write_throughput	\
	     --metrics shared_load_throughput	\
	     --metrics shared_store_throughput	\
	     --metrics stall_exec_dependency	\
	     --metrics stall_memory_dependency	\
	     --metrics stall_inst_fetch	 \
	     --metrics stall_other	 \
	     --metrics stall_pipe_bus 	 \
	     --metrics stall_pipe_busy	 \
	     --metrics inst_fp_32  \
	     --metrics inst_integer	 \
	     --metrics inst_control	 \
	     --metrics inst_compute_ld_st \
	     --metrics inst_executed	\
		./nbody ${N} ${DT} 1  0  $(INPUT) $(OUTPUT) 

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
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /weight
	#-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_x
	#-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_y
	#-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_z
	#-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_w

