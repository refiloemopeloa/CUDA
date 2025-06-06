#include "hitable_list_cpu.h"

#include <random>

using namespace std;

struct hit_record_cpu_cpu;

 float schlick_cpu(float cosine, float ref_idx) {
    float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*pow((1.0f - cosine),5.0f);
}

 bool refract(const vec3_cpu& v, const vec3_cpu& n, float ni_over_nt, vec3_cpu& refracted) {
    vec3_cpu uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > 0) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
        return false;
}

 vec3_cpu random_in_unit_sphere(uniform_real_distribution<float> &local_rand_state, default_random_engine &generator) {
    vec3_cpu p;
    do {
        p = 2.0f * vec3_cpu(local_rand_state(generator), local_rand_state(generator), local_rand_state(generator)) - vec3_cpu(1, 1, 1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

 vec3_cpu reflect(const vec3_cpu& v, const vec3_cpu& n) {
    return v - 2 * dot(v, n) * n;
}

class material_cpu {
    public:
          virtual bool scatter(const Ray_cpu &r_in, const hit_record_cpu &rec, vec3_cpu &attenuation, Ray_cpu &scattered, uniform_real_distribution<float> &local_rand_state, default_random_engine &generator) const = 0;
};

class lambertian_cpu : public material_cpu {
    public:
         lambertian_cpu(const vec3_cpu &a) : albedo(a) {}

         virtual bool scatter(const Ray_cpu &r_in, const hit_record_cpu &rec, vec3_cpu &attenuation, Ray_cpu &scattered, uniform_real_distribution<float> &local_rand_state, default_random_engine &generator) const {
            vec3_cpu target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state, generator);
            scattered = Ray_cpu(rec.p, target - rec.p);
            attenuation = albedo;
            return true;
        }

        vec3_cpu albedo;

};

class metal_cpu : public material_cpu {
    public:
         metal_cpu(const vec3_cpu &a, float f) : albedo(a) {
            if (f < 1.0f) {
                fuzz = f;
            } else {
                fuzz = 1.0f;
            }
        }

         virtual bool scatter(const Ray_cpu &r_in, const hit_record_cpu &rec, vec3_cpu &attenuation, Ray_cpu &scattered, uniform_real_distribution<float> &local_rand_state, default_random_engine &generator) const {
            vec3_cpu reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = Ray_cpu(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state, generator));
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }

    vec3_cpu albedo;
    float fuzz;
};

class dielectric_cpu : public material_cpu {
public:
     dielectric_cpu(float ri) : ref_idx(ri) {}

     virtual bool scatter(const Ray_cpu& r_in,
                         const hit_record_cpu& rec,
                         vec3_cpu& attenuation,
                         Ray_cpu& scattered,
                         uniform_real_distribution<float> &local_rand_state, 
                         default_random_engine &generator) const  {
        vec3_cpu outward_normal;
        vec3_cpu reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = vec3_cpu(1.0, 1.0, 1.0);
        vec3_cpu refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction(), rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = sqrt(1.0f - ref_idx*ref_idx*(1-cosine*cosine));
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick_cpu(cosine, ref_idx);
        else
            reflect_prob = 1.0f;
        if (local_rand_state(generator) < reflect_prob)
            scattered = Ray_cpu(rec.p, reflected);
        else
            scattered = Ray_cpu(rec.p, refracted);
        return true;
    }

    float ref_idx;
};