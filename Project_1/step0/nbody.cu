/**
 * @File nbody.cu
 *
 * Implementation of the N-Body problem
 *
 * Paralelní programování na GPU (PCG 2021)
 * Projekt c. 1 (cuda)
 * Login: xducho07
 */

#include <cmath>
#include <cfloat>
#include "nbody.h"

/**
 * CUDA kernel to calculate gravitation velocity
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_gravitation_velocity(t_particles p, t_velocities tmp_vel, int N, float dt)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N) {
        const float pos_x = p.pos_x[i];
        const float pos_y = p.pos_y[i];
        const float pos_z = p.pos_z[i];

        float r2, dx, dy, dz;
        float vx = 0.f, vy = 0.f, vz = 0.f;

        for (int j = 0; j < N; j++) {
            dx = pos_x - p.pos_x[j];
            dy = pos_y - p.pos_y[j];
            dz = pos_z - p.pos_z[j];

            r2 = dx*dx + dy*dy + dz*dz;

            if (r2 > COLLISION_DISTANCE * COLLISION_DISTANCE) {
                /* MISTO PRO VAS KOD GRAVITACE */

                //Version 2: Use 10 operations, math simplification

                // Fg*dt/m1/r = G*m1*m2*dt / r^3 / m1 = G*dt/r^3 * m2
                const float Fg_dt_m2_r = (-G * dt / (sqrtf(r2) * r2) + FLT_MIN) * p.weight[j];

                // vx = - Fx*dt/m2 = - Fg*dt/m2 * dx/r = - Fg*dt/m2/r * dx
                vx += Fg_dt_m2_r * dx;
                vy += Fg_dt_m2_r * dy;
                vz += Fg_dt_m2_r * dz;
            }
        }
        /* KONEC */

        tmp_vel.x[i] += vx;
        tmp_vel.y[i] += vy;
        tmp_vel.z[i] += vz;
    }
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate collision velocity
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_collision_velocity(t_particles p, t_velocities tmp_vel, int N, float dt)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N) {
        const float pos_x = p.pos_x[i];
        const float pos_y = p.pos_y[i];
        const float pos_z = p.pos_z[i];
        const float vel_x = p.vel_x[i];
        const float vel_y = p.vel_y[i];
        const float vel_z = p.vel_z[i];
        const float weight = p.weight[i];

        float r2, dx, dy, dz;
        float vx = vel_x, vy = vel_y, vz = vel_z;

        for (int j = 0; j < N; j++) {
            dx = pos_x - p.pos_x[j];
            dy = pos_y - p.pos_y[j];
            dz = pos_z - p.pos_z[j];

            r2 = dx*dx + dy*dy + dz*dz;

            /* MISTO PRO VAS KOD KOLIZE */
            if (r2 <= COLLISION_DISTANCE * COLLISION_DISTANCE && r2 > 0.f) {
                const float weight2 = p.weight[j];

                vx += ((weight * vel_x - weight2 * vel_x + 2 * weight2 * p.vel_x[j]) /
                       (weight + weight2)) - vel_x ;
                vy += ((weight * vel_y - weight2 * vel_y + 2 * weight2 * p.vel_y[j]) /
                       (weight + weight2)) - vel_y ;
                vz += ((weight * vel_z - weight2 * vel_z + 2 * weight2 * p.vel_z[j]) /
                       (weight + weight2)) - vel_z ;
            }
            /* KONEC */
        }

        tmp_vel.x[i] = vx;
        tmp_vel.y[i] = vy;
        tmp_vel.z[i] = vz;
    }
}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to update particles
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void update_particle(t_particles p, t_velocities tmp_vel, int N, float dt)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N) {
        const float vel_x = tmp_vel.x[i];
        const float vel_y = tmp_vel.y[i];
        const float vel_z = tmp_vel.z[i];

        p.vel_x[i] = vel_x;
        p.vel_y[i] = vel_y;
        p.vel_z[i] = vel_z;

        p.pos_x[i] += vel_x * dt;
        p.pos_y[i] += vel_y * dt;
        p.pos_z[i] += vel_z * dt;
    }
}// end of update_particle
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to update particles
 * @param p       - particles
 * @param comX    - pointer to a center of mass position in X
 * @param comY    - pointer to a center of mass position in Y
 * @param comZ    - pointer to a center of mass position in Z
 * @param comW    - pointer to a center of mass weight
 * @param lock    - pointer to a user-implemented lock
 * @param N       - Number of particles
 */
__global__ void centerOfMass(t_particles p, float* comX, float* comY, float* comZ, float* comW, int* lock, const int N)
{

}// end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------

/**
 * CPU implementation of the Center of Mass calculation
 * @param particles - All particles in the system
 * @param N         - Number of particles
 */
__host__ float4 centerOfMassCPU(MemDesc& memDesc)
{
  float4 com = {0 ,0, 0, 0};

  for(int i = 0; i < memDesc.getDataSize(); i++)
  {
    // Calculate the vector on the line connecting current body and most recent position of center-of-mass
    const float dx = memDesc.getPosX(i) - com.x;
    const float dy = memDesc.getPosY(i) - com.y;
    const float dz = memDesc.getPosZ(i) - com.z;

    // Calculate weight ratio only if at least one particle isn't massless
    const float dw = ((memDesc.getWeight(i) + com.w) > 0.0f)
                          ? ( memDesc.getWeight(i) / (memDesc.getWeight(i) + com.w)) : 0.0f;

    // Update position and weight of the center-of-mass according to the weight ration and vector
    com.x += dx * dw;
    com.y += dy * dw;
    com.z += dz * dw;
    com.w += memDesc.getWeight(i);
  }
  return com;
}// enf of centerOfMassCPU
//----------------------------------------------------------------------------------------------------------------------
