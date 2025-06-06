#include "sphere_cpu.h"

#define HITABLE_LIST
#ifdef HITABLE_LIST

class hitable_list_cpu : public hitable_cpu {
public:
    hitable_list_cpu() {}
    hitable_list_cpu(hitable_cpu **l, int n) { list = l; list_size = n; }

    virtual bool hit(const Ray_cpu& r, float t_min, float t_max, hit_record_cpu& rec) const;

    hitable_cpu **list;
    int list_size;
};

bool hitable_list_cpu::hit(const Ray_cpu& r, float t_min, float t_max, hit_record_cpu& rec) const {
    hit_record_cpu temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

#endif 