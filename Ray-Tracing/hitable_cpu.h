#include "ray_cpu.h"

#define HITABLE
#ifdef HITABLE

class material_cpu; // Forward declaration

struct hit_record_cpu {
    float t;
    vec3_cpu p;
    vec3_cpu normal;
    material_cpu* mat_ptr;
};

class hitable_cpu {
public:
    virtual bool hit(const Ray_cpu& r, float t_min, float t_max, hit_record_cpu& rec) const = 0;
};

#endif

