#include <iostream>
#include <chrono>
#include <fstream>
#include <string>

using namespace std;
using namespace std::chrono;

void render(int x, int y, int* pixels) {
    float r;
    float g;
    float b;
    int pixel_index;

    for (int j = y-1; j >= 0; j--) {
        for (int i = 0;  i < x; i++) {
            pixel_index = j * x * 3 + i * 3;
            r = float(i) / float(x);
            g = float(j) / float(y);
            b = 0.2;
            pixels[pixel_index + 0] = int(255.99 * r);
            pixels[pixel_index + 1] = int(255.99 * g);
            pixels[pixel_index + 2] = int(255.99 * b);
        }
    }
}

void print_image(int x, int y, int* pixels, ostream& file) {
    file << "P3\n" << x << " " << y << "\n255\n";

    for (int j = y-1; j >= 0; j--) {
        for (int i = 0;  i < x; i++) {
            int pixel_index = j * x * 3 + i * 3;
            file << pixels[pixel_index + 0] << " " << pixels[pixel_index + 1] << " " << pixels[pixel_index + 2] << "\n";
        }
    }
}

int main() {

    int x = 20000;
    int y = x/2;
    int *pixels = (int*)malloc(x * y * 3 * sizeof(int));
    string filename = "out-cpu-array.ppm";
    
    auto start = high_resolution_clock::now();
    render(x, y, pixels);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(stop - start);
    cout << "Time taken by function: " << duration.count() << " nanoseconds" << endl;
    
    ofstream file(filename);
    print_image(x, y, pixels, file);

    file.close();

    free(pixels);
    
    return 0;
}