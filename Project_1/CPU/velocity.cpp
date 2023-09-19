/*
 * Architektura procesoru (ACH 2016)
 * Projekt c. 1 (nbody)
 * Login: xducho07
 */

#include <cmath>
#include <cfloat>
#include "velocity.h"

/**
 * @breef   Funkce vypocte rychlost kterou teleso p1 ziska vlivem gravitace p2.
 * @details Viz zadani.pdf
 */
void calculate_gravitation_velocity(
  const t_particle &p1,
  const t_particle &p2,
  t_velocity &vel
)
{
    float vx, vy, vz;

    const float dx = p1.pos_x - p2.pos_x;
    const float dy = p1.pos_y - p2.pos_y;
    const float dz = p1.pos_z - p2.pos_z;

    const float r2 = dx*dx + dy*dy + dz*dz;

    /* MISTO PRO VAS KOD GRAVITACE */

    //Version 1: Many divisions, 22 operation
    //Force
    /*float F = -G * p1.weight * p2.weight / (r * r + FLT_MIN);
    //normalize per dimension
    float nx = F * dx/ (r + FLT_MIN);
    float ny = F * dy/ (r + FLT_MIN);
    float nz = F * dz/ (r + FLT_MIN);

    vx = nx * DT / p1.weight;
    vy = ny * DT / p1.weight;
    vz = nz * DT / p1.weight;*/

    if (r2 > COLLISION_DISTANCE * COLLISION_DISTANCE) {
        
        //Version 2: Use 10 operations, math simplification
        //
        // Fg*dt/m1/r = G*m1*m2*dt / r^3 / m1 = G*dt/r^3 * m2
        const float Fg_dt_m2_r = (-G * DT / (sqrtf(r2) * r2) + FLT_MIN) * p2.weight;
        
        // vx = - Fx*dt/m2 = - Fg*dt/m2 * dx/r = - Fg*dt/m2/r * dx
        vel.x += Fg_dt_m2_r * dx;
        vel.y += Fg_dt_m2_r * dy;
        vel.z += Fg_dt_m2_r * dz;
    }
}

/**
 * @breef   Funkce vypocte rozdil mezi rychlostmi pred a po kolizi telesa p1 do telesa p2.
 */
void calculate_collision_velocity(
  const t_particle &p1,
  const t_particle &p2,
  t_velocity &vel
)
{
    const float dx = p1.pos_x - p2.pos_x;
    const float dy = p1.pos_y - p2.pos_y;
    const float dz = p1.pos_z - p2.pos_z;

    const float r2 = dx*dx + dy*dy + dz*dz;

    /* MISTO PRO VAS KOD KOLIZE */

    /* KONEC */

    // jedna se o rozdilne ale blizke prvky
    if (r2 > 0.0f && r2 < COLLISION_DISTANCE * COLLISION_DISTANCE) {
        const float tmp = 1.f / (p1.weight + p2.weight);

        vel.x += ((p1.weight* p1.vel_x - p2.weight *p1.vel_x + 2* p2.weight* p2.vel_x) *
            tmp) - p1.vel_x ;
        vel.y += ((p1.weight* p1.vel_y - p2.weight *p1.vel_y + 2* p2.weight* p2.vel_y) *
            tmp) - p1.vel_y ;
        vel.z += ((p1.weight* p1.vel_z - p2.weight *p1.vel_z + 2* p2.weight* p2.vel_z) *
            tmp) - p1.vel_z ;
    }
}
