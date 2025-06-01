#include <iostream>

using namespace std;

void render(int x, int y) {
    cout << "P3\n" << x << " " << y << "\n255\n";
    for (int j = y-1; j >= 0; j--) {
        for (int i = 0;  i < x; i++) {
            float r = float(i) / float(x);
            float g = float(j) / float(y);
            float b = 0.2;
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);
            cout << ir << " " << ig << " " << ib << "\n";
        }
    }
}

int main() {
    render(200, 100);

    return 0;
}