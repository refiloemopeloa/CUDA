#include <cmath>

class vec3_cpu {
    
    public:
        vec3_cpu() {}
        vec3_cpu(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
        inline float x() const { return e[0]; }
        inline float y() const { return e[1]; }
        inline float z() const { return e[2]; }
        inline float r() const { return e[0]; }
        inline float g() const { return e[1]; }
        inline float b() const { return e[2]; }

        inline const vec3_cpu& operator+() const { return *this; }
        inline vec3_cpu operator-() const { return vec3_cpu(-e[0], -e[1], -e[2]); }
        inline float operator[](int i) const { return e[i]; }
        inline float& operator[](int i) { return e[i]; };

        inline vec3_cpu& operator+=(const vec3_cpu &v2);
        inline vec3_cpu& operator-=(const vec3_cpu &v2);
        inline vec3_cpu& operator*=(const vec3_cpu &v2);
        inline vec3_cpu& operator/=(const vec3_cpu &v2);
        inline vec3_cpu& operator*=(const float t);
        inline vec3_cpu& operator/=(const float t);

        inline float length() const {
            return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
        }
        inline float squared_length() const {
            return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
        }
        inline void make_unit_vector();

        float e[3];


        

};

inline vec3_cpu operator+(const vec3_cpu &v1, const vec3_cpu &v2) {
    return vec3_cpu(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}
inline vec3_cpu operator-(const vec3_cpu &v1, const vec3_cpu &v2) {
    return vec3_cpu(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}
inline vec3_cpu operator*(const vec3_cpu &v1, const vec3_cpu &v2) {
    return vec3_cpu(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}
inline vec3_cpu operator/(const vec3_cpu &v1, const vec3_cpu &v2) {
    return vec3_cpu(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}
inline vec3_cpu operator*(const vec3_cpu &v1, const float t) {
    return vec3_cpu(v1.e[0] * t, v1.e[1] * t, v1.e[2] * t);
}
inline vec3_cpu operator*(const float t, const vec3_cpu &v1) {
    return vec3_cpu(v1.e[0] * t, v1.e[1] * t, v1.e[2] * t);
}
inline vec3_cpu operator/(const vec3_cpu &v1, const float t) {
    return vec3_cpu(v1.e[0] / t, v1.e[1] / t, v1.e[2] / t);
}

// inline vec3_cpu operator/(const float t, const vec3_cpu &v1) {
//     return vec3_cpu(v1.e[0] / t, v1.e[1] / t, v1.e[2] / t);
// }

inline float dot(const vec3_cpu &v1, const vec3_cpu &v2) {
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}
inline vec3_cpu cross(const vec3_cpu &v1, const vec3_cpu &v2) {
    return vec3_cpu(v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1],
    v1.e[2] * v2.e[0] - v1.e[0] * v2.e[2],
    v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]);
}

inline vec3_cpu& vec3_cpu::operator+=(const vec3_cpu &v2) {
    e[0] += v2.e[0];
    e[1] += v2.e[1];
    e[2] += v2.e[2];
    return *this;
}

inline vec3_cpu& vec3_cpu::operator/=(const float t) {
    e[0] /= t;
    e[1] /= t;
    e[2] /= t;
    return *this;
}

inline vec3_cpu& vec3_cpu::operator*=(const vec3_cpu &v){
    e[0]  *= v.e[0];
    e[1]  *= v.e[1];
    e[2]  *= v.e[2];
    return *this;
}

inline vec3_cpu& vec3_cpu::operator/=(const vec3_cpu &v){
    e[0]  /= v.e[0];
    e[1]  /= v.e[1];
    e[2]  /= v.e[2];
    return *this;
}

inline vec3_cpu& vec3_cpu::operator-=(const vec3_cpu& v) {
    e[0]  -= v.e[0];
    e[1]  -= v.e[1];
    e[2]  -= v.e[2];
    return *this;
}

inline vec3_cpu& vec3_cpu::operator*=(const float t) {
    e[0]  *= t;
    e[1]  *= t;
    e[2]  *= t;
    return *this;
}


inline vec3_cpu unit_vector(vec3_cpu v) {
    return v / v.length();
};

inline void vec3_cpu::make_unit_vector() {
    float k = 1.0 / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    e[0] *= k; e[1] *= k; e[2] *= k;
}