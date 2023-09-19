/**
 * @file      nbody.cpp
 *
 * @author    Radek Duchon
 *            Faculty of Information Technology
 *            Brno University of Technology
 *            xducho07@stud.fit.vutbr.cz
 *
 * @brief     PCG Assignment 2
 *            N-Body simulation in ACC
 *
 * @version   2021
 *
 * @date      11 November  2020, 11:22 (created) \n
 * @date      15 November  2021, 14:10 (revised) \n
 *
 */

#include <math.h>
#include <cfloat>
#include "nbody.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                       Declare following structs / classes                                          //
//                                  If necessary, add your own classes / routines                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Compute gravitation velocity
void calculate_gravitation_velocity(const Particles& p,
                                    Velocities&      tmp_vel,
                                    const int        N,
                                    const float      dt)
{
    #pragma acc parallel loop present(p, tmp_vel) gang worker vector
    for (unsigned int i = 0; i < N; ++i) {
        const float pos_x = p.pos_x[i];
        const float pos_y = p.pos_y[i];
        const float pos_z = p.pos_z[i];

        float r2, dx, dy, dz;
        float vx = 0.f, vy = 0.f, vz = 0.f;

        #pragma acc loop seq
        for (int j = 0; j < N; j++) {
            dx = pos_x - p.pos_x[j];
            dy = pos_y - p.pos_y[j];
            dz = pos_z - p.pos_z[j];

            r2 = dx*dx + dy*dy + dz*dz;

            if (r2 > COLLISION_DISTANCE * COLLISION_DISTANCE) {
                const float Fg_dt_m2_r = (-G * dt / (sqrtf(r2) * r2) + FLT_MIN) * p.weight[j];
                vx += Fg_dt_m2_r * dx;
                vy += Fg_dt_m2_r * dy;
                vz += Fg_dt_m2_r * dz;
            }
        }

        tmp_vel.vel_x[i] += vx;
        tmp_vel.vel_y[i] += vy;
        tmp_vel.vel_z[i] += vz;
    }
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

void calculate_collision_velocity(const Particles& p,
                                  Velocities&      tmp_vel,
                                  const int        N,
                                  const float      dt)
{
    #pragma acc parallel loop present(p, tmp_vel) gang vector
    for (unsigned int i = 0; i < N; ++i) {
        const float pos_x = p.pos_x[i];
        const float pos_y = p.pos_y[i];
        const float pos_z = p.pos_z[i];
        const float vel_x = p.vel_x[i];
        const float vel_y = p.vel_y[i];
        const float vel_z = p.vel_z[i];
        const float weight = p.weight[i];

        float r2, dx, dy, dz;
        float vx = vel_x, vy = vel_y, vz = vel_z;

        #pragma acc loop seq
        for (int j = 0; j < N; j++) {
            dx = pos_x - p.pos_x[j];
            dy = pos_y - p.pos_y[j];
            dz = pos_z - p.pos_z[j];

            r2 = dx*dx + dy*dy + dz*dz;

            if (r2 <= COLLISION_DISTANCE * COLLISION_DISTANCE && r2 > 0.f) {
                const float weight2 = p.weight[j];

                vx += ((weight * vel_x - weight2 * vel_x + 2 * weight2 * p.vel_x[j]) /
                       (weight + weight2)) - vel_x;
                vy += ((weight * vel_y - weight2 * vel_y + 2 * weight2 * p.vel_y[j]) /
                       (weight + weight2)) - vel_y;
                vz += ((weight * vel_z - weight2 * vel_z + 2 * weight2 * p.vel_z[j]) /
                       (weight + weight2)) - vel_z;
            }
        }

        tmp_vel.vel_x[i] = vx;
        tmp_vel.vel_y[i] = vy;
        tmp_vel.vel_z[i] = vz;
    }
}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/// Update particle position
void update_particle(const Particles& p,
                     Velocities&      tmp_vel,
                     const int        N,
                     const float      dt)
{
    #pragma acc parallel loop present(p, tmp_vel) gang worker vector
    for (unsigned int i = 0; i < N; ++i) {
        const float vel_x = tmp_vel.vel_x[i];
        const float vel_y = tmp_vel.vel_y[i];
        const float vel_z = tmp_vel.vel_z[i];

        p.vel_x[i] = vel_x;
        p.vel_y[i] = vel_y;
        p.vel_z[i] = vel_z;

        p.pos_x[i] += vel_x * dt;
        p.pos_y[i] += vel_y * dt;
        p.pos_z[i] += vel_z * dt;
    }
}// end of update_particle
//----------------------------------------------------------------------------------------------------------------------



/// Compute center of gravity
float4 centerOfMassGPU(const Particles& p,
                       const int        N)
{

  return {0.0f, 0.0f, 0.0f, 0.0f};
}// end of centerOfMassGPU
//----------------------------------------------------------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Compute center of mass on CPU
float4 centerOfMassCPU(MemDesc& memDesc)
{
  float4 com = {0 ,0, 0, 0};

  for(int i = 0; i < memDesc.getDataSize(); i++)
  {
    // Calculate the vector on the line connecting points and most recent position of center-of-mass
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
}// end of centerOfMassCPU
//----------------------------------------------------------------------------------------------------------------------
