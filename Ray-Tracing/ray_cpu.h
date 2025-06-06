#include "vec3_cpu.h"

class Ray_cpu {
    public:
        Ray_cpu() {}
        Ray_cpu(const vec3_cpu& a, const vec3_cpu& b) { A = a; B = b; }
        vec3_cpu origin() const { return A; }
        vec3_cpu direction() const { return B; }
        vec3_cpu point_at_parameter(float t) const {
            return A + B*t;
        }

        vec3_cpu A;
        vec3_cpu B;
};