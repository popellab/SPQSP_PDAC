/**
 * QSP standalone simulator.
 *
 * Runs the PDAC QSP ODE system from a parameter XML and writes the species
 * trajectory to disk. Supports two output formats:
 *
 *   - CSV (human-readable, SimBiology-comparable). Column 0 is Time (days),
 *     remaining columns are species values in original (source) units.
 *   - Raw binary (compact, fast to parse from Python). Used for parameter
 *     sweeps where CSV parse overhead dominates wall time.
 *
 * Binary format (little-endian, packed, no padding):
 *   uint32  magic    = 0x51535042          // "QSPB"
 *   uint32  version  = 1
 *   uint64  n_times                         // number of time rows (incl. t=0)
 *   uint64  n_species                       // columns per row
 *   float64 dt_days
 *   float64 t_end_days
 *   float64 data[n_times * n_species]       // row-major; row = timepoint
 *
 * Time column is NOT stored — it is reconstructible as i*dt for i in
 * [0, n_times), with the final row being t_end_days (possibly clipped).
 *
 * Usage:
 *   qsp_sim --param <xml> [--csv-out <path>] [--binary-out <path>]
 *           [--species-out <path>] [--t-end-days N] [--dt-days N]
 *
 * Legacy positional form (kept for back-compat with existing tests):
 *   qsp_sim <param_xml> <csv_out> [t_end_days] [dt_days]
 */
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "qsp/ode/ODE_system.h"
#include "qsp/ode/QSPParam.h"
#include "qsp/ode/QSP_enum.h"

using namespace CancerVCT;

namespace {

struct Args {
    std::string param_file;
    std::string csv_out;
    std::string binary_out;
    std::string species_out;
    double t_end_days = 365.0;
    double dt_days = 0.1;
};

void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog
        << " --param <xml> [--csv-out <path>] [--binary-out <path>]\n"
        << "                  [--species-out <path>] [--t-end-days N] [--dt-days N]\n"
        << "\n"
        << "Legacy: " << prog
        << " <param_xml> <csv_out> [t_end_days] [dt_days]\n";
}

bool parse_args(int argc, char* argv[], Args& out) {
    if (argc >= 3 && argv[1][0] != '-') {
        out.param_file = argv[1];
        out.csv_out = argv[2];
        if (argc > 3) out.t_end_days = std::stod(argv[3]);
        if (argc > 4) out.dt_days = std::stod(argv[4]);
        return true;
    }

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_val = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << name << " requires a value" << std::endl;
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--param") {
            const char* v = need_val("--param"); if (!v) return false;
            out.param_file = v;
        } else if (a == "--csv-out") {
            const char* v = need_val("--csv-out"); if (!v) return false;
            out.csv_out = v;
        } else if (a == "--binary-out") {
            const char* v = need_val("--binary-out"); if (!v) return false;
            out.binary_out = v;
        } else if (a == "--species-out") {
            const char* v = need_val("--species-out"); if (!v) return false;
            out.species_out = v;
        } else if (a == "--t-end-days") {
            const char* v = need_val("--t-end-days"); if (!v) return false;
            out.t_end_days = std::stod(v);
        } else if (a == "--dt-days") {
            const char* v = need_val("--dt-days"); if (!v) return false;
            out.dt_days = std::stod(v);
        } else if (a == "-h" || a == "--help") {
            return false;
        } else {
            std::cerr << "Unknown argument: " << a << std::endl;
            return false;
        }
    }

    if (out.param_file.empty()) {
        std::cerr << "--param is required" << std::endl;
        return false;
    }
    if (out.csv_out.empty() && out.binary_out.empty()) {
        std::cerr << "At least one of --csv-out or --binary-out is required"
                  << std::endl;
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

#ifdef MODEL_UNITS
    const double time_factor = 1.0;
#else
    const double time_factor = 86400.0;
#endif
    const double t_end = args.t_end_days * time_factor;
    const double dt = args.dt_days * time_factor;

    QSPParam param;
    param.initializeParams(args.param_file);
    ODE_system::setup_class_parameters(param);

    ODE_system ode;
    ode.setup_instance_variables(param);
    ode.setup_instance_tolerance(param);
    ode.eval_init_assignment();

    const std::string header = ODE_system::getHeader();
    const size_t n_species = static_cast<size_t>(ode.getNumOutputSpecies());

    if (!args.species_out.empty()) {
        std::ofstream sp_out(args.species_out);
        size_t start = 0;
        for (size_t i = 0; i <= header.size(); ++i) {
            if (i == header.size() || header[i] == ',') {
                sp_out << header.substr(start, i - start) << '\n';
                start = i + 1;
            }
        }
    }

    std::ofstream csv;
    if (!args.csv_out.empty()) {
        csv.open(args.csv_out);
        csv << std::scientific << std::setprecision(12);
        // operator<< on CVODEBase emits all species prefixed with commas; pair
        // it with "Time,<header>" (no leading comma in getHeader()) for a
        // self-consistent CSV.
        csv << "Time," << header << std::endl;
    }

    // Binary header is written up front with a placeholder n_times; we seek
    // back and patch it after the stepping loop, since the loop condition
    // `t < t_end` with floating-point dt can produce one more or one fewer
    // step than ceil(t_end/dt) predicts.
    std::ofstream bin;
    const uint32_t MAGIC = 0x51535042u;  // "QSPB"
    const uint32_t VERSION = 1;
    if (!args.binary_out.empty()) {
        bin.open(args.binary_out, std::ios::binary);
        uint64_t n_times_placeholder = 0;
        uint64_t n_sp64 = static_cast<uint64_t>(n_species);
        bin.write(reinterpret_cast<const char*>(&MAGIC), sizeof(MAGIC));
        bin.write(reinterpret_cast<const char*>(&VERSION), sizeof(VERSION));
        bin.write(reinterpret_cast<const char*>(&n_times_placeholder), sizeof(uint64_t));
        bin.write(reinterpret_cast<const char*>(&n_sp64), sizeof(uint64_t));
        bin.write(reinterpret_cast<const char*>(&args.dt_days), sizeof(double));
        bin.write(reinterpret_cast<const char*>(&args.t_end_days), sizeof(double));
    }

    std::vector<double> row(n_species);

    auto write_state = [&](double t) {
        if (csv.is_open()) {
            csv << t / time_factor << ode;
            csv << std::endl;
        }
        if (bin.is_open()) {
            for (size_t i = 0; i < n_species; ++i) {
                row[i] = ode.getSpeciesOutputValue(static_cast<int>(i));
            }
            bin.write(reinterpret_cast<const char*>(row.data()),
                      static_cast<std::streamsize>(n_species * sizeof(double)));
        }
    };

    // Sampling-run mode: CVODE keeps its internal history across output
    // points (via simOdeSample) so fine output cadence doesn't force
    // the solver to repeatedly restart from a tiny step-size. This is
    // event-free — adequate for baseline scenarios; dosing scenarios
    // with explicit SBML events will need a step-based path.
    ode.setupSamplingRun(t_end);

    write_state(0.0);
    uint64_t n_times = 1;

    double t = 0.0;
    int step = 0;
    while (t < t_end) {
        t += dt;
        if (t > t_end) t = t_end;  // clamp last step to exact boundary
        ode.simOdeSample(t);
        step++;
        write_state(t);
        n_times++;

        if (step % 1000 == 0) {
            std::cerr << "  t=" << t / time_factor << " days" << std::endl;
        }
    }

    if (csv.is_open()) {
        csv.close();
        std::cerr << "Wrote " << n_times << " time points to " << args.csv_out
                  << std::endl;
    }
    if (bin.is_open()) {
        bin.seekp(sizeof(uint32_t) * 2, std::ios::beg);
        bin.write(reinterpret_cast<const char*>(&n_times), sizeof(uint64_t));
        bin.close();
        std::cerr << "Wrote " << n_times << " time points × " << n_species
                  << " species to " << args.binary_out << std::endl;
    }
    return 0;
}