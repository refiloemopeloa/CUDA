#include "hitable_cpu.h"

#define SPHERE
#ifdef SPHERE

class sphere_cpu : public hitable_cpu {
public:
    sphere_cpu() {}
    sphere_cpu(vec3_cpu cen, float r, material_cpu *m) : center(cen), radius(r), mat_ptr(m) {}

    virtual bool hit(const Ray_cpu& r, float t_min, float t_max, hit_record_cpu& rec) const;

    vec3_cpu center;
    float radius;
    material_cpu* mat_ptr;
};

bool sphere_cpu::hit(const Ray_cpu& r, float t_min, float t_max, hit_record_cpu& rec) const {
    vec3_cpu oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;

    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr; // Assign material pointer
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr; // Assign material pointer
            return true;
        }
    }
    return false;
}

#endif