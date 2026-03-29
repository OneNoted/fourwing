// Compile with: nvcc -x cu fourwing7.cpp -O3 -std=c++17 -o fourwing7

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kSeedAxisCount = 4;
constexpr std::uint32_t kSeedCount = kSeedAxisCount * kSeedAxisCount * kSeedAxisCount;
constexpr std::uint32_t kProgressPercentStep = 10;

enum class AttractorSystem : std::uint32_t {
    Fourwing = 1,
    Chen = 2,
};

enum class Backend : std::uint32_t {
    Auto = 0,
    Cuda = 1,
    Cpu = 2,
};

enum class Axis : std::uint32_t {
    A = 0,
    B = 1,
    C = 2,
};

struct AxisSpec {
    double min = 0.0;
    double max = 0.0;
    std::uint32_t count = 1;
};

struct SliceRequest {
    Axis axis = Axis::A;
    std::uint32_t index = 0;
};

struct SweepConfig {
    AttractorSystem system = AttractorSystem::Fourwing;
    Backend backend = Backend::Auto;
    AxisSpec a;
    AxisSpec b;
    AxisSpec c;
    double dt = 1.0 / 60.0;
    std::uint32_t max_steps = 100000;
    double escape_radius = 100.0;
    std::string out_prefix;
    std::optional<std::string> dump_volume_path;
    std::vector<SliceRequest> slice_requests;
    std::uint32_t seed_axis_count = kSeedAxisCount;
};

struct ParsedArgs {
    std::optional<AttractorSystem> system;
    std::optional<Backend> backend;
    std::optional<double> a_min;
    std::optional<double> a_max;
    std::optional<std::uint32_t> a_count;
    std::optional<double> b_min;
    std::optional<double> b_max;
    std::optional<std::uint32_t> b_count;
    std::optional<double> c_min;
    std::optional<double> c_max;
    std::optional<std::uint32_t> c_count;
    std::optional<double> dt;
    std::optional<std::uint32_t> max_steps;
    std::optional<double> escape_radius;
    std::optional<std::string> out_prefix;
    std::optional<std::string> dump_volume_path;
    std::vector<SliceRequest> slice_requests;
    bool help = false;
};

struct SliceMetadata {
    std::uint32_t width = 1;
    std::uint32_t height = 1;
    const char* row_axis = "a";
    const char* col_axis = "b";
};

struct InitialSeeds {
    std::array<double, kSeedCount> x{};
    std::array<double, kSeedCount> y{};
    std::array<double, kSeedCount> z{};
};

#pragma pack(push, 1)
struct VolumeDumpHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t system;
    std::uint32_t a_count;
    std::uint32_t b_count;
    std::uint32_t c_count;
    double a_min;
    double a_max;
    double b_min;
    double b_max;
    double c_min;
    double c_max;
    double dt;
    std::uint32_t max_steps;
    double escape_radius;
    std::uint32_t seed_axis_count;
    char layout[16];
    char reserved[16];
};
#pragma pack(pop)

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        const cudaError_t status__ = (call);                                                  \
        if (status__ != cudaSuccess) {                                                        \
            std::ostringstream oss__;                                                         \
            oss__ << "CUDA error: " << cudaGetErrorString(status__) << " at " << __FILE__    \
                  << ":" << __LINE__;                                                         \
            throw std::runtime_error(oss__.str());                                            \
        }                                                                                     \
    } while (false)

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

constexpr double kSeedStart = -0.3;
constexpr double kSeedStep = 0.2;

constexpr const char* system_name(AttractorSystem system) {
    return system == AttractorSystem::Fourwing ? "fourwing" : "chen";
}

constexpr const char* backend_name(Backend backend) {
    switch (backend) {
        case Backend::Auto:
            return "auto";
        case Backend::Cuda:
            return "cuda";
        case Backend::Cpu:
            return "cpu";
    }
    return "unknown";
}

constexpr char axis_code(Axis axis) {
    switch (axis) {
        case Axis::A:
            return 'a';
        case Axis::B:
            return 'b';
        case Axis::C:
            return 'c';
    }
    return '?';
}

constexpr const char* axis_name(Axis axis) {
    switch (axis) {
        case Axis::A:
            return "a";
        case Axis::B:
            return "b";
        case Axis::C:
            return "c";
    }
    return "?";
}

constexpr InitialSeeds make_initial_seeds() {
    InitialSeeds seeds{};
    for (std::uint32_t index = 0; index < kSeedCount; ++index) {
        const std::uint32_t ix = index & 0x3u;
        const std::uint32_t iy = (index >> 2u) & 0x3u;
        const std::uint32_t iz = (index >> 4u) & 0x3u;
        seeds.x[index] = kSeedStart + static_cast<double>(ix) * kSeedStep;
        seeds.y[index] = kSeedStart + static_cast<double>(iy) * kSeedStep;
        seeds.z[index] = kSeedStart + static_cast<double>(iz) * kSeedStep;
    }
    return seeds;
}

constexpr InitialSeeds kInitialSeeds = make_initial_seeds();

HD inline double axis_value(const AxisSpec& axis, std::uint32_t index) {
    if (axis.count <= 1) {
        return axis.min;
    }
    const double fraction = static_cast<double>(index) / static_cast<double>(axis.count - 1);
    return axis.min + (axis.max - axis.min) * fraction;
}

HD inline void decode_seed(std::uint32_t seed_index, double& x, double& y, double& z) {
    const std::uint32_t ix = seed_index & 0x3u;
    const std::uint32_t iy = (seed_index >> 2u) & 0x3u;
    const std::uint32_t iz = (seed_index >> 4u) & 0x3u;
    x = kSeedStart + static_cast<double>(ix) * kSeedStep;
    y = kSeedStart + static_cast<double>(iy) * kSeedStep;
    z = kSeedStart + static_cast<double>(iz) * kSeedStep;
}

template <AttractorSystem System>
HD inline void evaluate_field(double x, double y, double z, double a, double b, double c,
                              double& dx, double& dy, double& dz) {
    if constexpr (System == AttractorSystem::Fourwing) {
        dx = a * x + y * z;
        dy = b * x + c * y - x * z;
        dz = -z - x * y;
    } else {
        dx = a * (y - x);
        dy = (c - a) * x - x * z + c * y;
        dz = x * y - b * z;
    }
}

template <AttractorSystem System>
HD inline void rk4_step(double& x, double& y, double& z, double a, double b, double c,
                        double dt) {
    const double half_dt = 0.5 * dt;
    const double dt_over_six = dt / 6.0;

    double k1x = 0.0, k1y = 0.0, k1z = 0.0;
    evaluate_field<System>(x, y, z, a, b, c, k1x, k1y, k1z);

    const double x1 = x + half_dt * k1x;
    const double y1 = y + half_dt * k1y;
    const double z1 = z + half_dt * k1z;
    double k2x = 0.0, k2y = 0.0, k2z = 0.0;
    evaluate_field<System>(x1, y1, z1, a, b, c, k2x, k2y, k2z);

    const double x2 = x + half_dt * k2x;
    const double y2 = y + half_dt * k2y;
    const double z2 = z + half_dt * k2z;
    double k3x = 0.0, k3y = 0.0, k3z = 0.0;
    evaluate_field<System>(x2, y2, z2, a, b, c, k3x, k3y, k3z);

    const double x3 = x + dt * k3x;
    const double y3 = y + dt * k3y;
    const double z3 = z + dt * k3z;
    double k4x = 0.0, k4y = 0.0, k4z = 0.0;
    evaluate_field<System>(x3, y3, z3, a, b, c, k4x, k4y, k4z);

    x += dt_over_six * (k1x + 2.0 * (k2x + k3x) + k4x);
    y += dt_over_six * (k1y + 2.0 * (k2y + k3y) + k4y);
    z += dt_over_six * (k1z + 2.0 * (k2z + k3z) + k4z);
}

inline std::string to_lower_copy(std::string_view value) {
    std::string lowered(value);
    for (char& ch : lowered) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return lowered;
}

[[noreturn]] void usage_error(const std::string& message) {
    throw std::runtime_error(message);
}

std::string consume_value(const std::string& option_name, const std::optional<std::string>& inline_value,
                          int argc, char** argv, int& index) {
    if (inline_value.has_value()) {
        return *inline_value;
    }
    if (index + 1 >= argc) {
        usage_error("Missing value for " + option_name);
    }
    ++index;
    return argv[index];
}

double parse_double_value(const std::string& option_name, const std::string& raw) {
    std::size_t parsed = 0;
    const double value = std::stod(raw, &parsed);
    if (parsed != raw.size()) {
        usage_error("Invalid floating-point value for " + option_name + ": " + raw);
    }
    return value;
}

std::uint32_t parse_uint32_value(const std::string& option_name, const std::string& raw) {
    std::size_t parsed = 0;
    const unsigned long value = std::stoul(raw, &parsed, 10);
    if (parsed != raw.size()) {
        usage_error("Invalid integer value for " + option_name + ": " + raw);
    }
    if (value == 0 || value > std::numeric_limits<std::uint32_t>::max()) {
        usage_error("Out-of-range integer value for " + option_name + ": " + raw);
    }
    return static_cast<std::uint32_t>(value);
}

AttractorSystem parse_system(const std::string& raw) {
    const std::string lowered = to_lower_copy(raw);
    if (lowered == "fourwing") {
        return AttractorSystem::Fourwing;
    }
    if (lowered == "chen") {
        return AttractorSystem::Chen;
    }
    usage_error("Unknown system: " + raw);
}

Backend parse_backend(const std::string& raw) {
    const std::string lowered = to_lower_copy(raw);
    if (lowered == "auto") {
        return Backend::Auto;
    }
    if (lowered == "cuda") {
        return Backend::Cuda;
    }
    if (lowered == "cpu") {
        return Backend::Cpu;
    }
    usage_error("Unknown backend: " + raw);
}

Axis parse_axis(const std::string& raw) {
    const std::string lowered = to_lower_copy(raw);
    if (lowered == "a") {
        return Axis::A;
    }
    if (lowered == "b") {
        return Axis::B;
    }
    if (lowered == "c") {
        return Axis::C;
    }
    usage_error("Unknown axis: " + raw);
}

SliceRequest parse_slice_request(const std::string& raw) {
    const std::size_t separator = raw.find(':');
    if (separator == std::string::npos) {
        usage_error("Invalid slice specification: " + raw);
    }
    SliceRequest request;
    request.axis = parse_axis(raw.substr(0, separator));
    request.index = parse_uint32_value("--slice", raw.substr(separator + 1));
    return request;
}

void print_usage(std::ostream& out) {
    out << "Usage: fourwing7 --system {fourwing|chen} [options]\n"
        << "\n"
        << "Options:\n"
        << "  --backend {auto|cuda|cpu}        Execution backend, default auto\n"
        << "  --a-min <value>                  Parameter a range minimum\n"
        << "  --a-max <value>                  Parameter a range maximum\n"
        << "  --a-count <value>                Parameter a samples, default 128\n"
        << "  --b-min <value>                  Parameter b range minimum\n"
        << "  --b-max <value>                  Parameter b range maximum\n"
        << "  --b-count <value>                Parameter b samples, default 128\n"
        << "  --c-min <value>                  Parameter c range minimum\n"
        << "  --c-max <value>                  Parameter c range maximum\n"
        << "  --c-count <value>                Parameter c samples, default 128\n"
        << "  --dt <value>                     Integration step size, default 1/60\n"
        << "  --max-steps <value>              Maximum integration steps, default 100000\n"
        << "  --escape-radius <value>          Breakout radius, default 100\n"
        << "  --out-prefix <path>              Slice output prefix, default fourwing7_<system>\n"
        << "  --slice <axis:index>             Additional slice to export, repeatable\n"
        << "  --dump-volume <path>             Optional raw volume dump\n"
        << "  --help                           Show this help text\n"
        << "\n"
        << "Examples:\n"
        << "  ./fourwing7 --system chen\n"
        << "  ./fourwing7 --system fourwing --c-count 1 --c-min -0.4 --c-max -0.4\n"
        << "  ./fourwing7 --system chen --backend cpu --a-count 8 --b-count 8 --c-count 8\n";
}

ParsedArgs parse_args(int argc, char** argv) {
    ParsedArgs parsed{};

    for (int index = 1; index < argc; ++index) {
        std::string argument = argv[index];
        if (argument == "--help") {
            parsed.help = true;
            continue;
        }
        if (argument.rfind("--", 0) != 0) {
            usage_error("Unexpected positional argument: " + argument);
        }

        std::optional<std::string> inline_value;
        const std::size_t equal_sign = argument.find('=');
        if (equal_sign != std::string::npos) {
            inline_value = argument.substr(equal_sign + 1);
            argument = argument.substr(0, equal_sign);
        }

        if (argument == "--system") {
            parsed.system = parse_system(consume_value(argument, inline_value, argc, argv, index));
        } else if (argument == "--backend") {
            parsed.backend = parse_backend(consume_value(argument, inline_value, argc, argv, index));
        } else if (argument == "--a-min") {
            parsed.a_min = parse_double_value(argument, consume_value(argument, inline_value, argc, argv, index));
        } else if (argument == "--a-max") {
            parsed.a_max = parse_double_value(argument, consume_value(argument, inline_value, argc, argv, index));
        } else if (argument == "--a-count") {
            parsed.a_count = parse_uint32_value(argument, consume_value(argument, inline_value, argc, argv, index));
        } else if (argument == "--b-min") {
            parsed.b_min = parse_double_value(argument, consume_value(argument, inline_value, argc, argv, index));
        } else if (argument == "--b-max") {
            parsed.b_max = parse_double_value(argument, consume_value(argument, inline_value, argc, argv, index));
        } else if (argument == "--b-count") {
            parsed.b_count = parse_uint32_value(argument, consume_value(argument, inline_value, argc, argv, index));
        } else if (argument == "--c-min") {
            parsed.c_min = parse_double_value(argument, consume_value(argument, inline_value, argc, argv, index));
        } else if (argument == "--c-max") {
            parsed.c_max = parse_double_value(argument, consume_value(argument, inline_value, argc, argv, index));
        } else if (argument == "--c-count") {
            parsed.c_count = parse_uint32_value(argument, consume_value(argument, inline_value, argc, argv, index));
        } else if (argument == "--dt") {
            parsed.dt = parse_double_value(argument, consume_value(argument, inline_value, argc, argv, index));
        } else if (argument == "--max-steps") {
            parsed.max_steps = parse_uint32_value(argument, consume_value(argument, inline_value, argc, argv, index));
        } else if (argument == "--escape-radius") {
            parsed.escape_radius =
                parse_double_value(argument, consume_value(argument, inline_value, argc, argv, index));
        } else if (argument == "--out-prefix") {
            parsed.out_prefix = consume_value(argument, inline_value, argc, argv, index);
        } else if (argument == "--dump-volume") {
            parsed.dump_volume_path = consume_value(argument, inline_value, argc, argv, index);
        } else if (argument == "--slice") {
            parsed.slice_requests.push_back(
                parse_slice_request(consume_value(argument, inline_value, argc, argv, index)));
        } else {
            usage_error("Unknown option: " + argument);
        }
    }

    return parsed;
}

SweepConfig make_default_config(AttractorSystem system) {
    SweepConfig config{};
    config.system = system;
    config.backend = Backend::Auto;
    config.dt = 1.0 / 60.0;
    config.max_steps = 100000;
    config.escape_radius = 100.0;
    config.seed_axis_count = kSeedAxisCount;
    config.out_prefix = std::string("fourwing7_") + system_name(system);

    config.a = {0.0, 14.0, 128};
    config.b = {-200.0, -2.0, 128};
    config.c = {-9.5, -0.1, 128};

    if (system == AttractorSystem::Fourwing) {
        config.a = {-1.8, 2.2, 128};
        config.b = {-0.09, 0.11, 128};
        config.c = {3.6, -4.4, 128};
    }

    return config;
}

SweepConfig build_config(const ParsedArgs& parsed) {
    if (!parsed.system.has_value()) {
        usage_error("Missing required option: --system");
    }

    SweepConfig config = make_default_config(*parsed.system);

    if (parsed.backend.has_value()) {
        config.backend = *parsed.backend;
    }
    if (parsed.a_min.has_value()) {
        config.a.min = *parsed.a_min;
    }
    if (parsed.a_max.has_value()) {
        config.a.max = *parsed.a_max;
    }
    if (parsed.a_count.has_value()) {
        config.a.count = *parsed.a_count;
    }
    if (parsed.b_min.has_value()) {
        config.b.min = *parsed.b_min;
    }
    if (parsed.b_max.has_value()) {
        config.b.max = *parsed.b_max;
    }
    if (parsed.b_count.has_value()) {
        config.b.count = *parsed.b_count;
    }
    if (parsed.c_min.has_value()) {
        config.c.min = *parsed.c_min;
    }
    if (parsed.c_max.has_value()) {
        config.c.max = *parsed.c_max;
    }
    if (parsed.c_count.has_value()) {
        config.c.count = *parsed.c_count;
    }
    if (parsed.dt.has_value()) {
        config.dt = *parsed.dt;
    }
    if (parsed.max_steps.has_value()) {
        config.max_steps = *parsed.max_steps;
    }
    if (parsed.escape_radius.has_value()) {
        config.escape_radius = *parsed.escape_radius;
    }
    if (parsed.out_prefix.has_value()) {
        config.out_prefix = *parsed.out_prefix;
    }
    if (parsed.dump_volume_path.has_value()) {
        config.dump_volume_path = *parsed.dump_volume_path;
    }
    config.slice_requests = parsed.slice_requests;

    if (config.dt <= 0.0) {
        usage_error("--dt must be positive");
    }
    if (config.escape_radius <= 0.0) {
        usage_error("--escape-radius must be positive");
    }

    return config;
}

std::uint64_t total_cells(const SweepConfig& config) {
    const std::uint64_t ab = static_cast<std::uint64_t>(config.a.count) * config.b.count;
    return ab * config.c.count;
}

std::size_t safe_total_cells(const SweepConfig& config) {
    const std::uint64_t total = total_cells(config);
    if (total == 0) {
        usage_error("Volume dimensions must be non-zero");
    }
    if (total > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t))) {
        usage_error("Requested volume is too large for this build");
    }
    return static_cast<std::size_t>(total);
}

std::size_t volume_index(const SweepConfig& config, std::uint32_t ia, std::uint32_t ib, std::uint32_t ic) {
    return (static_cast<std::size_t>(ia) * config.b.count + ib) * config.c.count + ic;
}

template <AttractorSystem System>
std::uint32_t simulate_cell_cpu(double a, double b, double c, const SweepConfig& config) {
    alignas(64) double x[kSeedCount];
    alignas(64) double y[kSeedCount];
    alignas(64) double z[kSeedCount];

    std::memcpy(x, kInitialSeeds.x.data(), sizeof(x));
    std::memcpy(y, kInitialSeeds.y.data(), sizeof(y));
    std::memcpy(z, kInitialSeeds.z.data(), sizeof(z));

    const double escape_radius2 = config.escape_radius * config.escape_radius;
    for (std::uint32_t step = 0; step < config.max_steps; ++step) {
        bool escaped = false;
        for (std::uint32_t seed = 0; seed < kSeedCount; ++seed) {
            rk4_step<System>(x[seed], y[seed], z[seed], a, b, c, config.dt);
            const double radius2 = x[seed] * x[seed] + y[seed] * y[seed] + z[seed] * z[seed];
            escaped = escaped || (radius2 > escape_radius2);
        }
        if (escaped) {
            return step;
        }
    }

    return config.max_steps;
}

void maybe_report_progress(std::uint32_t completed, std::uint32_t total, int& last_reported,
                           const Clock::time_point& start_time, std::string_view label) {
    const int percent_complete = static_cast<int>((static_cast<std::uint64_t>(completed) * 100) / total);
    if (percent_complete < last_reported + static_cast<int>(kProgressPercentStep) &&
        percent_complete < 100) {
        return;
    }

    const auto now = Clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
    const double estimated_total = elapsed * 100.0 / std::max(1, percent_complete);
    const double remaining = std::max(0.0, estimated_total - elapsed);

    std::cout << "[" << label << "] " << percent_complete << "% (" << completed << "/" << total
              << ") | Elapsed: " << elapsed << "s | ETA: " << remaining << "s" << std::endl;
    last_reported = percent_complete;
}

template <AttractorSystem System>
void run_cpu_sweep(const SweepConfig& config, std::vector<std::uint32_t>& volume) {
    const auto start_time = Clock::now();
    int last_reported = -static_cast<int>(kProgressPercentStep);

    for (std::uint32_t ia = 0; ia < config.a.count; ++ia) {
        const double a = axis_value(config.a, ia);
        for (std::uint32_t ib = 0; ib < config.b.count; ++ib) {
            const double b = axis_value(config.b, ib);
            for (std::uint32_t ic = 0; ic < config.c.count; ++ic) {
                const double c = axis_value(config.c, ic);
                volume[volume_index(config, ia, ib, ic)] = simulate_cell_cpu<System>(a, b, c, config);
            }
        }
        maybe_report_progress(ia + 1, config.a.count, last_reported, start_time,
                              "volume sweep / cpu");
    }
}

template <AttractorSystem System>
__global__ void sweep_a_slice_kernel(std::uint32_t* volume, AxisSpec b_axis, AxisSpec c_axis,
                                     std::uint32_t a_index, double a_value, double dt,
                                     double escape_radius2, std::uint32_t max_steps,
                                     std::uint32_t c_count) {
    const std::uint32_t ic = blockIdx.x;
    const std::uint32_t ib = blockIdx.y;
    const std::uint32_t seed = threadIdx.x;

    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    decode_seed(seed, x, y, z);

    const double b_value = axis_value(b_axis, ib);
    const double c_value = axis_value(c_axis, ic);

    std::uint32_t step = 0;
    for (; step < max_steps; ++step) {
        rk4_step<System>(x, y, z, a_value, b_value, c_value, dt);
        const double radius2 = x * x + y * y + z * z;
        const int escaped_here = radius2 > escape_radius2;
        if (__syncthreads_or(escaped_here)) {
            break;
        }
    }

    if (seed == 0) {
        volume[(static_cast<std::size_t>(a_index) * b_axis.count + ib) * c_count + ic] = step;
    }
}

int first_cuda_device() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice) {
        cudaGetLastError();
        return -1;
    }
    CUDA_CHECK(status);
    return device_count > 0 ? 0 : -1;
}

template <AttractorSystem System>
void run_cuda_sweep(const SweepConfig& config, std::vector<std::uint32_t>& volume) {
    const int device_index = first_cuda_device();
    if (device_index < 0) {
        throw std::runtime_error("No CUDA device is available");
    }

    CUDA_CHECK(cudaSetDevice(device_index));

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device_index));
    std::cout << "Using CUDA device: " << properties.name << std::endl;

    std::uint32_t* device_volume = nullptr;
    CUDA_CHECK(cudaMalloc(&device_volume, volume.size() * sizeof(std::uint32_t)));

    const auto start_time = Clock::now();
    int last_reported = -static_cast<int>(kProgressPercentStep);

    try {
        for (std::uint32_t ia = 0; ia < config.a.count; ++ia) {
            const double a = axis_value(config.a, ia);
            const dim3 grid(config.c.count, config.b.count, 1);
            sweep_a_slice_kernel<System><<<grid, kSeedCount>>>(
                device_volume, config.b, config.c, ia, a, config.dt,
                config.escape_radius * config.escape_radius, config.max_steps, config.c.count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            maybe_report_progress(ia + 1, config.a.count, last_reported, start_time,
                                  "volume sweep / cuda");
        }

        CUDA_CHECK(cudaMemcpy(volume.data(), device_volume, volume.size() * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(device_volume);
        throw;
    }

    CUDA_CHECK(cudaFree(device_volume));
}

template <AttractorSystem System>
void run_sweep(Backend backend, const SweepConfig& config, std::vector<std::uint32_t>& volume) {
    if (backend == Backend::Cpu) {
        run_cpu_sweep<System>(config, volume);
        return;
    }

    if (backend == Backend::Cuda) {
        run_cuda_sweep<System>(config, volume);
        return;
    }

    const int device_index = first_cuda_device();
    if (device_index >= 0) {
        run_cuda_sweep<System>(config, volume);
        return;
    }

    std::cout << "No CUDA device detected, falling back to CPU backend." << std::endl;
    run_cpu_sweep<System>(config, volume);
}

Backend resolve_backend(Backend requested) {
    if (requested != Backend::Auto) {
        return requested;
    }
    return first_cuda_device() >= 0 ? Backend::Cuda : Backend::Cpu;
}

std::uint32_t midpoint_index(const AxisSpec& axis) {
    return axis.count / 2;
}

const AxisSpec& axis_spec(const SweepConfig& config, Axis axis) {
    switch (axis) {
        case Axis::A:
            return config.a;
        case Axis::B:
            return config.b;
        case Axis::C:
            return config.c;
    }
    return config.a;
}

std::vector<SliceRequest> collect_slice_requests(const SweepConfig& config) {
    std::vector<SliceRequest> requests = config.slice_requests;
    requests.push_back({Axis::A, midpoint_index(config.a)});
    requests.push_back({Axis::B, midpoint_index(config.b)});
    requests.push_back({Axis::C, midpoint_index(config.c)});

    for (const SliceRequest& request : requests) {
        if (request.index >= axis_spec(config, request.axis).count) {
            std::ostringstream oss;
            oss << "Slice index " << request.index << " is out of range for axis "
                << axis_name(request.axis);
            throw std::runtime_error(oss.str());
        }
    }

    std::sort(requests.begin(), requests.end(),
              [](const SliceRequest& lhs, const SliceRequest& rhs) {
                  if (lhs.axis != rhs.axis) {
                      return static_cast<std::uint32_t>(lhs.axis) <
                             static_cast<std::uint32_t>(rhs.axis);
                  }
                  return lhs.index < rhs.index;
              });
    requests.erase(std::unique(requests.begin(), requests.end(),
                               [](const SliceRequest& lhs, const SliceRequest& rhs) {
                                   return lhs.axis == rhs.axis && lhs.index == rhs.index;
                               }),
                   requests.end());
    return requests;
}

SliceMetadata slice_metadata(const SweepConfig& config, Axis axis) {
    switch (axis) {
        case Axis::A:
            return {config.c.count, config.b.count, "b", "c"};
        case Axis::B:
            return {config.c.count, config.a.count, "a", "c"};
        case Axis::C:
            return {config.b.count, config.a.count, "a", "b"};
    }
    return {1, 1, "a", "b"};
}

std::uint32_t slice_value(const std::vector<std::uint32_t>& volume, const SweepConfig& config,
                          Axis axis, std::uint32_t fixed_index, std::uint32_t row,
                          std::uint32_t column) {
    switch (axis) {
        case Axis::A:
            return volume[volume_index(config, fixed_index, row, column)];
        case Axis::B:
            return volume[volume_index(config, row, fixed_index, column)];
        case Axis::C:
            return volume[volume_index(config, row, column, fixed_index)];
    }
    return 0;
}

std::filesystem::path slice_output_path(const SweepConfig& config, Axis axis, std::uint32_t index) {
    const std::filesystem::path prefix_path(config.out_prefix);
    const std::filesystem::path parent = prefix_path.parent_path();
    const std::string stem =
        prefix_path.filename().empty() ? std::string("fourwing7") : prefix_path.filename().string();
    const std::size_t width =
        std::max<std::size_t>(1, std::to_string(axis_spec(config, axis).count - 1).size());

    std::ostringstream filename;
    filename << stem << "_" << axis_code(axis) << "_" << std::setw(static_cast<int>(width))
             << std::setfill('0') << index << ".pgm";
    return parent / filename.str();
}

void ensure_parent_directory(const std::filesystem::path& path) {
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

void write_pgm_slice(const std::vector<std::uint32_t>& volume, const SweepConfig& config,
                     const SliceRequest& request) {
    const SliceMetadata metadata = slice_metadata(config, request.axis);
    std::vector<unsigned char> pixels(static_cast<std::size_t>(metadata.width) * metadata.height);

    for (std::uint32_t row = 0; row < metadata.height; ++row) {
        for (std::uint32_t column = 0; column < metadata.width; ++column) {
            const std::uint32_t steps = slice_value(volume, config, request.axis, request.index, row, column);
            const double normalized =
                static_cast<double>(steps) / static_cast<double>(std::max<std::uint32_t>(1, config.max_steps));
            const auto pixel = static_cast<unsigned char>(
                std::llround(std::clamp(normalized, 0.0, 1.0) * 255.0));
            pixels[static_cast<std::size_t>(row) * metadata.width + column] = pixel;
        }
    }

    const std::filesystem::path path = slice_output_path(config, request.axis, request.index);
    ensure_parent_directory(path);

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open slice output: " + path.string());
    }

    file << "P5\n";
    file << "# system=" << system_name(config.system) << "\n";
    file << "# fixed_axis=" << axis_name(request.axis) << "\n";
    file << "# fixed_index=" << request.index << "\n";
    file << "# fixed_value=" << axis_value(axis_spec(config, request.axis), request.index) << "\n";
    file << "# rows=" << metadata.row_axis << "\n";
    file << "# cols=" << metadata.col_axis << "\n";
    file << "# dt=" << config.dt << "\n";
    file << "# max_steps=" << config.max_steps << "\n";
    file << "# escape_radius=" << config.escape_radius << "\n";
    file << metadata.width << " " << metadata.height << "\n255\n";
    file.write(reinterpret_cast<const char*>(pixels.data()),
               static_cast<std::streamsize>(pixels.size()));

    std::cout << "Wrote slice: " << path << std::endl;
}

void write_volume_dump(const std::vector<std::uint32_t>& volume, const SweepConfig& config,
                       const std::filesystem::path& path) {
    ensure_parent_directory(path);

    VolumeDumpHeader header{};
    std::memcpy(header.magic, "FW7VOL1", 8);
    header.version = 1;
    header.system = static_cast<std::uint32_t>(config.system);
    header.a_count = config.a.count;
    header.b_count = config.b.count;
    header.c_count = config.c.count;
    header.a_min = config.a.min;
    header.a_max = config.a.max;
    header.b_min = config.b.min;
    header.b_max = config.b.max;
    header.c_min = config.c.min;
    header.c_max = config.c.max;
    header.dt = config.dt;
    header.max_steps = config.max_steps;
    header.escape_radius = config.escape_radius;
    header.seed_axis_count = config.seed_axis_count;
    std::memcpy(header.layout, "a-major,b,c", 12);

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open volume dump output: " + path.string());
    }

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    file.write(reinterpret_cast<const char*>(volume.data()),
               static_cast<std::streamsize>(volume.size() * sizeof(std::uint32_t)));
    std::cout << "Wrote volume dump: " << path << std::endl;
}

void print_config_summary(const SweepConfig& config, Backend resolved_backend) {
    std::cout << "System: " << system_name(config.system) << "\n"
              << "Backend: " << backend_name(resolved_backend) << "\n"
              << "a range: [" << config.a.min << ", " << config.a.max << "] x " << config.a.count
              << "\n"
              << "b range: [" << config.b.min << ", " << config.b.max << "] x " << config.b.count
              << "\n"
              << "c range: [" << config.c.min << ", " << config.c.max << "] x " << config.c.count
              << "\n"
              << "dt: " << config.dt << "\n"
              << "max steps: " << config.max_steps << "\n"
              << "escape radius: " << config.escape_radius << "\n"
              << "seed lattice: " << config.seed_axis_count << " x " << config.seed_axis_count
              << " x " << config.seed_axis_count << "\n"
              << "out prefix: " << config.out_prefix << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const ParsedArgs parsed = parse_args(argc, argv);
        if (parsed.help) {
            print_usage(std::cout);
            return 0;
        }

        const SweepConfig config = build_config(parsed);
        const Backend backend = resolve_backend(config.backend);
        print_config_summary(config, backend);

        std::vector<std::uint32_t> volume(safe_total_cells(config));

        const auto start_time = Clock::now();
        if (config.system == AttractorSystem::Fourwing) {
            run_sweep<AttractorSystem::Fourwing>(backend, config, volume);
        } else {
            run_sweep<AttractorSystem::Chen>(backend, config, volume);
        }

        if (config.dump_volume_path.has_value()) {
            write_volume_dump(volume, config, *config.dump_volume_path);
        }

        for (const SliceRequest& request : collect_slice_requests(config)) {
            write_pgm_slice(volume, config, request);
        }

        const auto end_time = Clock::now();
        const auto seconds =
            std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        std::cout << "Total execution: " << seconds << " s" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        std::cerr << "Run with --help for usage." << std::endl;
        return 1;
    }
}
