// Compile with: g++ fourwing7.cpp -fopenmp -Ofast -march=native -flto -std=c++17 -o fourwing7

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <omp.h>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kGridSize = 100;
constexpr int kTotalJobs = kGridSize * kGridSize;
constexpr int kMaxSteps = 100000;

constexpr double kDt = 1.0 / 60.0;
constexpr double kHalfDt = kDt * 0.5;
constexpr double kDtOverSix = kDt / 6.0;
constexpr double kEscapeRadius2 = 100.0 * 100.0;

constexpr double kAStart = -1.8;
constexpr double kAStep = (2.2 - kAStart) / static_cast<double>(kGridSize - 1);
constexpr double kB = 0.01;
constexpr double kCStart = 3.6;
constexpr double kCStep = (-4.4 - kCStart) / static_cast<double>(kGridSize - 1);

// Preserve the current floating-step start grid exactly: {-0.3, -0.1, 0.1}.
constexpr std::array<double, 3> kAxisValues = {-0.3, -0.1, 0.1};

constexpr std::size_t countRepresentativeParticles() {
    std::size_t count = 0;
    for (double x : kAxisValues) {
        for (double y : kAxisValues) {
            for (double z : kAxisValues) {
                const bool has_partner = (x != -0.3) && (y != -0.3);
                if (!has_partner || x > 0.0) {
                    ++count;
                }
            }
        }
    }
    return count;
}

constexpr std::size_t kParticleCount = countRepresentativeParticles();
static_assert(kParticleCount == 21, "Unexpected representative particle count");

struct InitialState {
    std::array<double, kParticleCount> x{};
    std::array<double, kParticleCount> y{};
    std::array<double, kParticleCount> z{};
};

constexpr InitialState makeInitialState() {
    InitialState state{};
    std::size_t idx = 0;

    for (double x : kAxisValues) {
        for (double y : kAxisValues) {
            for (double z : kAxisValues) {
                // Keep one representative for the exact symmetry (x, y, z) <-> (-x, -y, z).
                const bool has_partner = (x != -0.3) && (y != -0.3);
                if (!has_partner || x > 0.0) {
                    state.x[idx] = x;
                    state.y[idx] = y;
                    state.z[idx] = z;
                    ++idx;
                }
            }
        }
    }

    return state;
}

constexpr InitialState kInitialState = makeInitialState();

inline int simulate_steps(double a, double c) noexcept {
    alignas(64) double x[kParticleCount];
    alignas(64) double y[kParticleCount];
    alignas(64) double z[kParticleCount];

    std::memcpy(x, kInitialState.x.data(), sizeof(x));
    std::memcpy(y, kInitialState.y.data(), sizeof(y));
    std::memcpy(z, kInitialState.z.data(), sizeof(z));

    for (int step = 0; step < kMaxSteps; ++step) {
        int escaped = 0;

        #pragma omp simd aligned(x, y, z : 64) reduction(| : escaped)
        for (std::size_t i = 0; i < kParticleCount; ++i) {
            const double x0 = x[i];
            const double y0 = y[i];
            const double z0 = z[i];

            const double k1x = a * x0 + y0 * z0;
            const double k1y = kB * x0 + c * y0 - x0 * z0;
            const double k1z = -z0 - x0 * y0;

            const double x1 = x0 + kHalfDt * k1x;
            const double y1 = y0 + kHalfDt * k1y;
            const double z1 = z0 + kHalfDt * k1z;
            const double k2x = a * x1 + y1 * z1;
            const double k2y = kB * x1 + c * y1 - x1 * z1;
            const double k2z = -z1 - x1 * y1;

            const double x2 = x0 + kHalfDt * k2x;
            const double y2 = y0 + kHalfDt * k2y;
            const double z2 = z0 + kHalfDt * k2z;
            const double k3x = a * x2 + y2 * z2;
            const double k3y = kB * x2 + c * y2 - x2 * z2;
            const double k3z = -z2 - x2 * y2;

            const double x3 = x0 + kDt * k3x;
            const double y3 = y0 + kDt * k3y;
            const double z3 = z0 + kDt * k3z;
            const double k4x = a * x3 + y3 * z3;
            const double k4y = kB * x3 + c * y3 - x3 * z3;
            const double k4z = -z3 - x3 * y3;

            const double next_x = x0 + kDtOverSix * (k1x + 2.0 * (k2x + k3x) + k4x);
            const double next_y = y0 + kDtOverSix * (k1y + 2.0 * (k2y + k3y) + k4y);
            const double next_z = z0 + kDtOverSix * (k1z + 2.0 * (k2z + k3z) + k4z);

            x[i] = next_x;
            y[i] = next_y;
            z[i] = next_z;

            escaped |= (next_x * next_x + next_y * next_y + next_z * next_z) > kEscapeRadius2;
        }

        if (escaped != 0) {
            return step;
        }
    }

    return kMaxSteps;
}

void saveMatrixToCSV(const std::array<int, kTotalJobs>& matrix, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error writing " << filename << '\n';
        return;
    }

    file << std::fixed << std::setprecision(0);
    for (int row = 0; row < kGridSize; ++row) {
        for (int col = 0; col < kGridSize; ++col) {
            file << matrix[row * kGridSize + col];
            if (col + 1 < kGridSize) {
                file << ',';
            }
        }
        file << '\n';
    }

    std::cout << "Saved " << filename << " successfully." << std::endl;
}

}  // namespace

int main() {
    const auto start_time = Clock::now();

    std::array<int, kTotalJobs> c_a_array{};
    std::atomic<int> completed_jobs{0};
    std::atomic<int> next_report{10};

    #pragma omp parallel for schedule(guided, 8)
    for (int job = 0; job < kTotalJobs; ++job) {
        const int ci = job / kGridSize;
        const int ai = job % kGridSize;

        const double a = kAStart + ai * kAStep;
        const double c = kCStart + ci * kCStep;
        c_a_array[job] = simulate_steps(a, c);

        const int completed = completed_jobs.fetch_add(1, std::memory_order_relaxed) + 1;
        const int percent_complete = (completed * 100) / kTotalJobs;

        int report_target = next_report.load(std::memory_order_relaxed);
        while (percent_complete >= report_target && report_target <= 100) {
            if (next_report.compare_exchange_weak(
                    report_target, report_target + 10, std::memory_order_relaxed)) {
                const auto now = Clock::now();
                const auto elapsed =
                    std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                const double est_total = elapsed * 100.0 / std::max(1, report_target);
                const double remaining = std::max(0.0, est_total - elapsed);

                #pragma omp critical
                {
                    std::cout << "[c_a matrix] " << report_target << "% ("
                              << completed << "/" << kTotalJobs << ") | "
                              << "Elapsed: " << elapsed << "s | "
                              << "ETA: " << remaining << "s" << std::endl;
                }
                break;
            }
        }
    }

    saveMatrixToCSV(c_a_array, "c_a_array.csv");

    const auto end_time = Clock::now();
    std::cout << "Total execution: "
              << std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count()
              << " s" << std::endl;

    return 0;
}
