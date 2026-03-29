// Compile with: g++ fourwing5.cpp -fopenmp -O3 -std=c++17 -o fourwing_parallel_ab

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <array>
#include <filesystem>
#include <string>

using namespace std;
using namespace std::chrono;

const double dt = 1.0 / 60.0;

array<double, 3> Chen(double x, double y, double z, double a, double b, double c) {
    return { a * (y - x), (c - a) * x - x * z + c * y, x * y - b * z };
}

class Particle {
public:
    double x, y, z;

    Particle(double x, double y, double z) : x(x), y(y), z(z) {}

    void RK4(double a, double b, double c) {
        auto f = [a, b, c](double x, double y, double z) {
            return Chen(x, y, z, a, b, c);
        };

        double x0 = x, y0 = y, z0 = z;
        auto k1 = f(x0, y0, z0);
        auto k2 = f(x0 + dt * 0.5 * k1[0], y0 + dt * 0.5 * k1[1], z0 + dt * 0.5 * k1[2]);
        auto k3 = f(x0 + dt * 0.5 * k2[0], y0 + dt * 0.5 * k2[1], z0 + dt * 0.5 * k2[2]);
        auto k4 = f(x0 + dt * k3[0], y0 + dt * k3[1], z0 + dt * k3[2]);

        x += dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6;
        y += dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6;
        z += dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6;
    }
};

vector<double> linspace(double start, double end, int num) {
    vector<double> result(num);
    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i)
        result[i] = start + i * step;
    return result;
}

void saveMatrixToCSV(const vector<vector<int>>& matrix, const string& filename) {
    ofstream file(filename);
    if (file.is_open()) {
        file << fixed << setprecision(0);
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                file << matrix[i][j];
                if (j < matrix[i].size() - 1) file << ",";
            }
            file << "\n";
        }
        file.close();
    } else {
        cerr << "Error writing " << filename << endl;
    }
}

int main() {
    auto start_time = steady_clock::now();

    // Create layers directory if it doesn't exist to prevent clutter
    string output_dir = "layers";
    if (!filesystem::exists(output_dir)) {
        filesystem::create_directory(output_dir);
    }

    int grid_size = 128;
    vector<double> a_range = linspace(0.0, 14.0, grid_size);
    vector<double> b_range = linspace(-200.0, -2.0, grid_size);
    vector<double> c_range = linspace(-9.5, -0.1, grid_size);

    int total_layers = grid_size;

    cout << "Starting Chen Attractor simulation over 128x128x128 grid..." << endl;
    cout << "Layers will be saved in the '" << output_dir << "/' directory." << endl;

    for (int ci = 0; ci < total_layers; ++ci) {
        double current_c = c_range[ci];
        vector<vector<int>> a_b_array(grid_size, vector<int>(grid_size, 0));
        
        int completed_steps = 0;
        int total_steps = grid_size * grid_size;
        int last_reported = -10;

        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int ai = 0; ai < grid_size; ++ai) {
            for (int bi = 0; bi < grid_size; ++bi) {
                vector<Particle> particles;
                for (double x = -0.3; x <= 0.3; x += 0.2)
                    for (double y = -0.3; y <= 0.3; y += 0.2)
                        for (double z = -0.3; z <= 0.3; z += 0.2)
                            particles.emplace_back(x, y, z);

                int steps = 0;
                for (; steps < 100000; ++steps) {
                    bool breakout = false;
                    for (auto& p : particles) {
                        p.RK4(a_range[ai], b_range[bi], current_c);
                        if (sqrt(p.x*p.x + p.y*p.y + p.z*p.z) > 1000000.0) { breakout = true; break; }
                    }
                    if (breakout) break;
                }
                a_b_array[ai][bi] = steps;

                #pragma omp atomic
                completed_steps++;

                int percent_complete = (completed_steps * 100) / total_steps;

                #pragma omp critical
                {
                    if (percent_complete >= last_reported + 10) {
                        last_reported = percent_complete;
                    }
                }
            }
        }

        // Save layer
        string filename = output_dir + "/layer_" + to_string(ci) + ".csv";
        saveMatrixToCSV(a_b_array, filename);

        auto now = steady_clock::now();
        auto elapsed = duration_cast<seconds>(now - start_time).count();
        
        cout << "[Progress] Completed layer " << ci << " / " << (total_layers - 1) 
             << " | Saved to " << filename << "\n"
             << "   -> Current c = " << current_c << "\n"
             << "   -> a range: [0.0, 14.0], b range: [-200.0, -2.0]\n"
             << "   -> Total time elapsed: " << elapsed << "s\n" << endl;
    }

    auto end_time = steady_clock::now();
    cout << "\nTotal execution: "
         << duration_cast<seconds>(end_time - start_time).count()
         << " s" << endl;

    return 0;
}
