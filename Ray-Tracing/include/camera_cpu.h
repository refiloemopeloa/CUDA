#define CAMERA
#ifdef CAMERA

#include "material_cpu.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif



 vec3_cpu random_in_unit_disk(uniform_real_distribution<float> &local_rand_state, 
                         default_random_engine &generator) {
    vec3_cpu p;
    do {
        p = 2.0f*vec3_cpu(local_rand_state(generator),local_rand_state(generator),0) - vec3_cpu(1,1,0);
    } while (dot(p,p) >= 1.0f);
    return p;
}

class camera_cpu {
    public:
    camera_cpu(vec3_cpu lookfrom, vec3_cpu lookat, vec3_cpu vup, float vfov, float aspect, float aperture, float focus_dist) { // vfov is top to bottom in degrees
        lens_radius = aperture / 2.0f;
        float theta = vfov*((float)M_PI)/180.0f;
        float half_height = tan(theta/2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin  - half_width*focus_dist*u -half_height*focus_dist*v - focus_dist*w;
        horizontal = 2.0f*half_width*focus_dist*u;
        vertical = 2.0f*half_height*focus_dist*v;
    }


         Ray_cpu get_ray(float u_f, float v_f, uniform_real_distribution<float> &local_rand_state, 
                         default_random_engine &generator) const {
            vec3_cpu rd = lens_radius * random_in_unit_disk(local_rand_state, generator);
            vec3_cpu offset = u * rd.x() + v * rd.y();
            vec3_cpu u_horizontal = u_f * horizontal;
            vec3_cpu v_vertical = v_f * vertical;
            return Ray_cpu(origin + offset, lower_left_corner + u_horizontal + v_vertical - origin - offset);
        }

        vec3_cpu lower_left_corner;
        vec3_cpu horizontal;
        vec3_cpu vertical;
        vec3_cpu origin;
        vec3_cpu u, v, w;
        float lens_radius;
};

#endif