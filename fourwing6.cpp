// Compile with: g++ fourwing6.cpp -fopenmp -O3 -I /path/to/eigen -o fourwing_parallel_bc

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace Eigen;
using namespace std::chrono;

const double dt = 1.0 / 60.0;

vector<double> Fourwing(double x, double y, double z, double a = 0.2, double b = 0.01, double c = -0.4) {
    return { a * x + y * z, b * x + c * y - x * z, -z - x * y };
}

class Particle {
public:
    double x, y, z;

    Particle(double x, double y, double z) : x(x), y(y), z(z) {}

    void RK4(double a, double b, double c) {
        auto f = [a, b, c](double x, double y, double z) {
            return Fourwing(x, y, z, a, b, c);
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

void saveMatrixToCSV(const MatrixXd& matrix, const string& filename) {
    ofstream file(filename);
    if (file.is_open()) {
        file << fixed << setprecision(0);
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                file << matrix(i, j);
                if (j < matrix.cols() - 1) file << ",";
            }
            file << "\n";
        }
        file.close();
        cout << "Saved " << filename << " successfully." << endl;
    } else {
        cerr << "Error writing " << filename << endl;
    }
}

int main() {
    auto start_time = steady_clock::now();

    vector<double> b_range = linspace(-0.09, 0.11, 100);
    vector<double> c_range = linspace(3.6, -4.4, 100);

    MatrixXd b_c_array(100, 100);
    int completed_steps = 0, total_steps = 10000, last_reported = -10;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int bi = 0; bi < 100; ++bi) {
        for (int ci = 0; ci < 100; ++ci) {
            vector<Particle> particles;
            for (double x = -0.3; x <= 0.3; x += 0.2)
                for (double y = -0.3; y <= 0.3; y += 0.2)
                    for (double z = -0.3; z <= 0.3; z += 0.2)
                        particles.emplace_back(x, y, z);

            int steps = 0;
            for (; steps < 100000; ++steps) {
                bool breakout = false;
                for (auto& p : particles) {
                    p.RK4(0.2, b_range[bi], c_range[ci]);
                    if (sqrt(p.x*p.x + p.y*p.y + p.z*p.z) > 100.0) { breakout = true; break; }
                }
                if (breakout) break;
            }
            b_c_array(bi, ci) = steps;

            #pragma omp atomic
            completed_steps++;

            int percent_complete = (completed_steps * 100) / total_steps;

            #pragma omp critical
            {
                if (percent_complete >= last_reported + 10) {
                    auto now = steady_clock::now();
                    auto elapsed = duration_cast<seconds>(now - start_time).count();
                    double est_total = elapsed * 100.0 / max(1, percent_complete);
                    double remaining = max(0.0, est_total - elapsed);

                    cout << "[b_c matrix] " << percent_complete << "% ("
                         << completed_steps << "/" << total_steps << ") | "
                         << "Elapsed: " << elapsed << "s | "
                         << "ETA: " << remaining << "s" << endl;

                    last_reported = percent_complete;
                }
            }
        }
    }

    saveMatrixToCSV(b_c_array, "b_c_array.csv");

    auto end_time = steady_clock::now();
    cout << "Total execution: "
         << duration_cast<seconds>(end_time - start_time).count()
         << " s" << endl;

    return 0;
}
