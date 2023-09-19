# mark_description "Intel(R) C++ Intel(R) 64 Compiler for applications running on Intel(R) 64, Version 18.0.5.274 Build 20180823";
# mark_description "";
# mark_description "-std=c++11 -lpapi -ansi-alias -O2 -Wall -DN=131072 -DDT=0.001f -DSTEPS=50 -S -fsource-asm -c";
	.file "nbody.cpp"
	.text
..TXTST0:
.L_2__routine_start__Z18particles_simulateRA131072_10t_particle_0:
# -- Begin  _Z18particles_simulateRA131072_10t_particle
	.text
# mark_begin;
       .align    16,0x90
	.globl _Z18particles_simulateRA131072_10t_particle
# --- particles_simulate(t_particles &)
_Z18particles_simulateRA131072_10t_particle:
# parameter 1: %rdi
..B1.1:                         # Preds ..B1.0
                                # Execution count [1.00e+00]

### {

	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
..___tag_value__Z18particles_simulateRA131072_10t_particle.1:
..L2:
                                                          #10.1
        pushq     %r12                                          #10.1
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
        pushq     %r13                                          #10.1
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
        pushq     %r14                                          #10.1
	.cfi_def_cfa_offset 32
	.cfi_offset 14, -32
        pushq     %r15                                          #10.1
	.cfi_def_cfa_offset 40
	.cfi_offset 15, -40
        pushq     %rbx                                          #10.1
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
        pushq     %rbp                                          #10.1
	.cfi_def_cfa_offset 56
	.cfi_offset 6, -56
        subq      $1572904, %rsp                                #10.1
	.cfi_def_cfa_offset 1572960
        movq      %rdi, %rbp                                    #10.1

###     int i, j, k;
### 
###     t_velocities velocities = {};

        xorl      %esi, %esi                                    #13.29
        lea       16(%rsp), %rdi                                #13.29
        movl      $1572864, %edx                                #13.29
        call      _intel_fast_memset                            #13.29
                                # LOE rbx rbp r12 r13 r14 r15
..B1.2:                         # Preds ..B1.1
                                # Execution count [1.00e+00]

### 
###     for (k = 0; k < STEPS; k++)
###     {
###             //vynulovani mezisouctu
###         for (i = 0; i < N; i++)
###         {
###             velocities[i].x = 0.0f;
###             velocities[i].y = 0.0f;
###             velocities[i].z = 0.0f;
###         }
###             //vypocet nove rychlosti
###         for (i = 0; i < N; i++)
###         {
###             for (j = 0; j < N; j++)
###             {
###                 calculate_gravitation_velocity(p[j], p[i], velocities[j]);
###                 calculate_collision_velocity(p[j], p[i], velocities[j]);
###             }
###         }
###             //ulozeni rychlosti a posun castic
###         for (i = 0; i < N; i++)
###         {
###             p[i].vel_x += velocities[i].x;
###             p[i].vel_y += velocities[i].y;
###             p[i].vel_z += velocities[i].z;
### 
###             p[i].pos_x += p[i].vel_x * DT;
###             p[i].pos_y += p[i].vel_y * DT;
###             p[i].pos_z += p[i].vel_z * DT;

        movss     .L_2il0floatpacket.12(%rip), %xmm0            #42.40
        xorl      %eax, %eax                                    #15.10
        xorb      %dl, %dl                                      #15.10
                                # LOE rbp eax dl xmm0
..B1.3:                         # Preds ..B1.13 ..B1.2
                                # Execution count [5.00e+01]
        movl      %eax, %ebx                                    #18.9
        xorl      %ecx, %ecx                                    #18.9
                                # LOE rcx rbp eax ebx dl xmm0
..B1.4:                         # Preds ..B1.4 ..B1.3
                                # Execution count [6.55e+06]
        incl      %ebx                                          #18.9
        movl      %eax, 16(%rsp,%rcx)                           #20.13
        movl      %eax, 20(%rsp,%rcx)                           #20.13
        movl      %eax, 24(%rsp,%rcx)                           #20.13
        addq      $12, %rcx                                     #18.9
        cmpl      $131072, %ebx                                 #18.9
        jb        ..B1.4        # Prob 99%                      #18.9
                                # LOE rcx rbp eax ebx dl xmm0
..B1.5:                         # Preds ..B1.4
                                # Execution count [5.00e+01]
        movb      %dl, (%rsp)                                   #25.14[spill]
        movl      %eax, %esi                                    #25.14
        xorl      %ecx, %ecx                                    #25.14
                                # LOE rcx rbp eax esi
..B1.6:                         # Preds ..B1.10 ..B1.5
                                # Execution count [6.55e+06]
        xorl      %ebx, %ebx                                    #27.18
        lea       (%rbp,%rcx), %rdx                             #29.54
        movq      %rdx, 1572888(%rsp)                           #27.18[spill]
        movl      %eax, %r12d                                   #27.18
        movq      %rcx, 1572880(%rsp)                           #27.18[spill]
        xorl      %r13d, %r13d                                  #27.18
        movl      %esi, 8(%rsp)                                 #27.18[spill]
                                # LOE rbx rbp r13 r12d
..B1.7:                         # Preds ..B1.9 ..B1.6
                                # Execution count [8.59e+11]
        movq      1572888(%rsp), %rsi                           #29.17[spill]
        lea       (%rbp,%rbx), %r14                             #29.48
        movq      %r14, %rdi                                    #29.17
        lea       16(%rsp,%r13), %r15                           #29.60
        movq      %r15, %rdx                                    #29.17
..___tag_value__Z18particles_simulateRA131072_10t_particle.21:
#       calculate_gravitation_velocity(const t_particle &, const t_particle &, t_velocity &)
        call      _Z30calculate_gravitation_velocityRK10t_particleS1_R10t_velocity #29.17
..___tag_value__Z18particles_simulateRA131072_10t_particle.22:
                                # LOE rbx rbp r13 r14 r15 r12d
..B1.8:                         # Preds ..B1.7
                                # Execution count [8.59e+11]
        movq      %r14, %rdi                                    #30.17
        movq      %r15, %rdx                                    #30.17
        movq      1572888(%rsp), %rsi                           #30.17[spill]
..___tag_value__Z18particles_simulateRA131072_10t_particle.23:
#       calculate_collision_velocity(const t_particle &, const t_particle &, t_velocity &)
        call      _Z28calculate_collision_velocityRK10t_particleS1_R10t_velocity #30.17
..___tag_value__Z18particles_simulateRA131072_10t_particle.24:
                                # LOE rbx rbp r13 r12d
..B1.9:                         # Preds ..B1.8
                                # Execution count [8.59e+11]
        incl      %r12d                                         #27.32
        addq      $28, %rbx                                     #27.32
        addq      $12, %r13                                     #27.32
        cmpl      $131072, %r12d                                #27.29
        jl        ..B1.7        # Prob 99%                      #27.29
                                # LOE rbx rbp r13 r12d
..B1.10:                        # Preds ..B1.9
                                # Execution count [6.55e+06]
        movl      8(%rsp), %esi                                 #[spill]
        xorl      %eax, %eax                                    #
        incl      %esi                                          #25.28
        movq      1572880(%rsp), %rcx                           #[spill]
        addq      $28, %rcx                                     #25.28
        cmpl      $131072, %esi                                 #25.25
        jl        ..B1.6        # Prob 99%                      #25.25
                                # LOE rcx rbp eax esi
..B1.11:                        # Preds ..B1.10
                                # Execution count [5.00e+01]
        xorl      %ebx, %ebx                                    #34.9
        movl      %eax, %esi                                    #34.9
        movss     .L_2il0floatpacket.12(%rip), %xmm0            #
        xorl      %ecx, %ecx                                    #34.9
        movb      (%rsp), %dl                                   #[spill]
        .align    16,0x90
                                # LOE rcx rbx rbp eax esi dl xmm0
..B1.12:                        # Preds ..B1.12 ..B1.11
                                # Execution count [6.55e+06]
        movss     20(%rbx,%rbp), %xmm1                          #38.13
        incl      %esi                                          #34.9
        movss     16(%rbx,%rbp), %xmm2                          #37.13
        movss     12(%rbx,%rbp), %xmm3                          #36.13
        addss     24(%rsp,%rcx), %xmm1                          #38.13
        addss     20(%rsp,%rcx), %xmm2                          #37.13
        addss     16(%rsp,%rcx), %xmm3                          #36.13
        movss     %xmm1, 20(%rbx,%rbp)                          #38.13
        addq      $12, %rcx                                     #34.9
        movss     %xmm2, 16(%rbx,%rbp)                          #37.13
        movss     %xmm3, 12(%rbx,%rbp)                          #36.13
        mulss     %xmm0, %xmm1                                  #42.40
        mulss     %xmm0, %xmm2                                  #41.40
        mulss     %xmm0, %xmm3                                  #40.40
        addss     8(%rbx,%rbp), %xmm1                           #42.13
        addss     4(%rbx,%rbp), %xmm2                           #41.13
        addss     (%rbx,%rbp), %xmm3                            #40.13
        movss     %xmm1, 8(%rbx,%rbp)                           #42.13
        movss     %xmm2, 4(%rbx,%rbp)                           #41.13
        movss     %xmm3, (%rbx,%rbp)                            #40.13
        addq      $28, %rbx                                     #34.9
        cmpl      $131072, %esi                                 #34.9
        jb        ..B1.12       # Prob 99%                      #34.9
                                # LOE rcx rbx rbp eax esi dl xmm0
..B1.13:                        # Preds ..B1.12
                                # Execution count [5.00e+01]
        incb      %dl                                           #15.28
        cmpb      $50, %dl                                      #15.21
        jl        ..B1.3        # Prob 98%                      #15.21
                                # LOE rbp eax dl xmm0
..B1.14:                        # Preds ..B1.13
                                # Execution count [1.00e+00]

###         }
###     }
### }

        addq      $1572904, %rsp                                #45.1
	.cfi_def_cfa_offset 56
	.cfi_restore 6
        popq      %rbp                                          #45.1
	.cfi_def_cfa_offset 48
	.cfi_restore 3
        popq      %rbx                                          #45.1
	.cfi_def_cfa_offset 40
	.cfi_restore 15
        popq      %r15                                          #45.1
	.cfi_def_cfa_offset 32
	.cfi_restore 14
        popq      %r14                                          #45.1
	.cfi_def_cfa_offset 24
	.cfi_restore 13
        popq      %r13                                          #45.1
	.cfi_def_cfa_offset 16
	.cfi_restore 12
        popq      %r12                                          #45.1
	.cfi_def_cfa_offset 8
        ret                                                     #45.1
        .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	_Z18particles_simulateRA131072_10t_particle,@function
	.size	_Z18particles_simulateRA131072_10t_particle,.-_Z18particles_simulateRA131072_10t_particle
..LN_Z18particles_simulateRA131072_10t_particle.0:
	.data
# -- End  _Z18particles_simulateRA131072_10t_particle
	.text
.L_2__routine_start__Z14particles_readP8_IO_FILERA131072_10t_particle_1:
# -- Begin  _Z14particles_readP8_IO_FILERA131072_10t_particle
	.text
# mark_begin;
       .align    16,0x90
	.globl _Z14particles_readP8_IO_FILERA131072_10t_particle
# --- particles_read(FILE *, t_particles &)
_Z14particles_readP8_IO_FILERA131072_10t_particle:
# parameter 1: %rdi
# parameter 2: %rsi
..B2.1:                         # Preds ..B2.0
                                # Execution count [1.00e+00]

### {

	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
..___tag_value__Z14particles_readP8_IO_FILERA131072_10t_particle.43:
..L44:
                                                         #49.1
        pushq     %r12                                          #49.1
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
        pushq     %r13                                          #49.1
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
        pushq     %r14                                          #49.1
	.cfi_def_cfa_offset 32
	.cfi_offset 14, -32

###     for (int i = 0; i < N; i++)

        xorl      %eax, %eax                                    #50.16
        movl      %eax, %r12d                                   #50.16
        movq      %rsi, %r14                                    #50.16
        movq      %rdi, %r13                                    #50.16
                                # LOE rbx rbp r13 r14 r15 r12d
..B2.2:                         # Preds ..B2.3 ..B2.1
                                # Execution count [1.31e+05]

###     {
###         fscanf(fp, "%f %f %f %f %f %f %f \n",

        addq      $-32, %rsp                                    #52.9
	.cfi_def_cfa_offset 64
        lea       4(%r14), %rcx                                 #52.9
        movq      %r13, %rdi                                    #52.9
        lea       8(%r14), %r8                                  #52.9
        movl      $.L_2__STRING.0, %esi                         #52.9
        lea       12(%r14), %r9                                 #52.9
        movq      %r14, %rdx                                    #52.9
        xorl      %eax, %eax                                    #52.9
        lea       16(%r14), %r10                                #52.9
        movq      %r10, (%rsp)                                  #52.9
        lea       20(%r14), %r11                                #52.9
        movq      %r11, 8(%rsp)                                 #52.9
        lea       24(%r14), %r10                                #52.9
        movq      %r10, 16(%rsp)                                #52.9
#       fscanf(FILE *, const char *, ...)
        call      fscanf                                        #52.9
                                # LOE rbx rbp r13 r14 r15 r12d
..B2.7:                         # Preds ..B2.2
                                # Execution count [1.31e+05]
        addq      $32, %rsp                                     #52.9
	.cfi_def_cfa_offset 32
                                # LOE rbx rbp r13 r14 r15 r12d
..B2.3:                         # Preds ..B2.7
                                # Execution count [1.31e+05]
        incl      %r12d                                         #50.28
        addq      $28, %r14                                     #50.28
        cmpl      $131072, %r12d                                #50.25
        jl        ..B2.2        # Prob 99%                      #50.25
                                # LOE rbx rbp r13 r14 r15 r12d
..B2.4:                         # Preds ..B2.3
                                # Execution count [1.00e+00]

###             &p[i].pos_x, &p[i].pos_y, &p[i].pos_z,
###             &p[i].vel_x, &p[i].vel_y, &p[i].vel_z,
###             &p[i].weight);
###     }
### }

	.cfi_restore 14
        popq      %r14                                          #57.1
	.cfi_def_cfa_offset 24
	.cfi_restore 13
        popq      %r13                                          #57.1
	.cfi_def_cfa_offset 16
	.cfi_restore 12
        popq      %r12                                          #57.1
	.cfi_def_cfa_offset 8
        ret                                                     #57.1
        .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	_Z14particles_readP8_IO_FILERA131072_10t_particle,@function
	.size	_Z14particles_readP8_IO_FILERA131072_10t_particle,.-_Z14particles_readP8_IO_FILERA131072_10t_particle
..LN_Z14particles_readP8_IO_FILERA131072_10t_particle.1:
	.data
# -- End  _Z14particles_readP8_IO_FILERA131072_10t_particle
	.text
.L_2__routine_start__Z15particles_writeP8_IO_FILERA131072_10t_particle_2:
# -- Begin  _Z15particles_writeP8_IO_FILERA131072_10t_particle
	.text
# mark_begin;
       .align    16,0x90
	.globl _Z15particles_writeP8_IO_FILERA131072_10t_particle
# --- particles_write(FILE *, t_particles &)
_Z15particles_writeP8_IO_FILERA131072_10t_particle:
# parameter 1: %rdi
# parameter 2: %rsi
..B3.1:                         # Preds ..B3.0
                                # Execution count [1.00e+00]

### {

	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
..___tag_value__Z15particles_writeP8_IO_FILERA131072_10t_particle.60:
..L61:
                                                         #60.1
        pushq     %r12                                          #60.1
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
        pushq     %r13                                          #60.1
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
        pushq     %r14                                          #60.1
	.cfi_def_cfa_offset 32
	.cfi_offset 14, -32
        pushq     %r15                                          #60.1
	.cfi_def_cfa_offset 40
	.cfi_offset 15, -40
        pushq     %rsi                                          #60.1
	.cfi_def_cfa_offset 48

###     for (int i = 0; i < N; i++)

        xorl      %edx, %edx                                    #61.16
        xorl      %eax, %eax                                    #61.16
        movl      %edx, %r13d                                   #61.16
        movq      %rax, %r12                                    #61.16
        movq      %rsi, %r15                                    #61.16
        movq      %rdi, %r14                                    #61.16
                                # LOE rbx rbp r12 r14 r15 r13d
..B3.2:                         # Preds ..B3.3 ..B3.1
                                # Execution count [1.31e+05]

###     {
###         fprintf(fp, "%10.10f %10.10f %10.10f %10.10f %10.10f %10.10f %10.10f \n",

        pxor      %xmm0, %xmm0                                  #63.9
        pxor      %xmm1, %xmm1                                  #63.9
        pxor      %xmm2, %xmm2                                  #63.9
        pxor      %xmm3, %xmm3                                  #63.9
        pxor      %xmm4, %xmm4                                  #63.9
        pxor      %xmm5, %xmm5                                  #63.9
        pxor      %xmm6, %xmm6                                  #63.9
        movq      %r14, %rdi                                    #63.9
        cvtss2sd  (%r12,%r15), %xmm0                            #63.9
        cvtss2sd  4(%r12,%r15), %xmm1                           #63.9
        cvtss2sd  8(%r12,%r15), %xmm2                           #63.9
        cvtss2sd  12(%r12,%r15), %xmm3                          #63.9
        cvtss2sd  16(%r12,%r15), %xmm4                          #63.9
        cvtss2sd  20(%r12,%r15), %xmm5                          #63.9
        cvtss2sd  24(%r12,%r15), %xmm6                          #63.9
        movl      $.L_2__STRING.1, %esi                         #63.9
        movl      $7, %eax                                      #63.9
#       fprintf(FILE *, const char *, ...)
        call      fprintf                                       #63.9
                                # LOE rbx rbp r12 r14 r15 r13d
..B3.3:                         # Preds ..B3.2
                                # Execution count [1.31e+05]
        incl      %r13d                                         #61.28
        addq      $28, %r12                                     #61.28
        cmpl      $131072, %r13d                                #61.25
        jl        ..B3.2        # Prob 99%                      #61.25
                                # LOE rbx rbp r12 r14 r15 r13d
..B3.4:                         # Preds ..B3.3
                                # Execution count [1.00e+00]

###             p[i].pos_x, p[i].pos_y, p[i].pos_z,
###             p[i].vel_x, p[i].vel_y, p[i].vel_z,
###             p[i].weight);
###     }
### }

        popq      %rcx                                          #68.1
	.cfi_def_cfa_offset 40
	.cfi_restore 15
        popq      %r15                                          #68.1
	.cfi_def_cfa_offset 32
	.cfi_restore 14
        popq      %r14                                          #68.1
	.cfi_def_cfa_offset 24
	.cfi_restore 13
        popq      %r13                                          #68.1
	.cfi_def_cfa_offset 16
	.cfi_restore 12
        popq      %r12                                          #68.1
	.cfi_def_cfa_offset 8
        ret                                                     #68.1
        .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	_Z15particles_writeP8_IO_FILERA131072_10t_particle,@function
	.size	_Z15particles_writeP8_IO_FILERA131072_10t_particle,.-_Z15particles_writeP8_IO_FILERA131072_10t_particle
..LN_Z15particles_writeP8_IO_FILERA131072_10t_particle.2:
	.data
# -- End  _Z15particles_writeP8_IO_FILERA131072_10t_particle
	.section .rodata, "a"
	.align 4
	.align 4
.L_2il0floatpacket.12:
	.long	0x3a83126f
	.type	.L_2il0floatpacket.12,@object
	.size	.L_2il0floatpacket.12,4
	.section .rodata.str1.4, "aMS",@progbits,1
	.align 4
	.align 4
.L_2__STRING.0:
	.long	622880293
	.long	1713709158
	.long	543565088
	.long	622880293
	.long	1713709158
	.word	2592
	.byte	0
	.type	.L_2__STRING.0,@object
	.size	.L_2__STRING.0,23
	.space 1, 0x00 	# pad
	.align 4
.L_2__STRING.1:
	.long	774910245
	.long	543567921
	.long	774910245
	.long	543567921
	.long	774910245
	.long	543567921
	.long	774910245
	.long	543567921
	.long	774910245
	.long	543567921
	.long	774910245
	.long	543567921
	.long	774910245
	.long	543567921
	.word	10
	.type	.L_2__STRING.1,@object
	.size	.L_2__STRING.1,58
	.data
	.section .note.GNU-stack, ""
// -- Begin DWARF2 SEGMENT .eh_frame
	.section .eh_frame,"a",@progbits
.eh_frame_seg:
	.align 8
# End
