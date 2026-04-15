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
 *           [--scenario <scenario.yaml> --drug-metadata <drug_meta.yaml>]
 *
 * With --scenario, doses declared in the scenario YAML are applied as boluses
 * to their target species between integration steps. This switches the solver
 * from the fast sampling path (setupSamplingRun + simOdeSample) to the
 * step-based path (simOdeStep), which is required for mid-integration state
 * perturbations.
 *
 * Legacy positional form (kept for back-compat with existing tests):
 *   qsp_sim <param_xml> <csv_out> [t_end_days] [dt_days]
 */
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

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
    std::string scenario_yaml;
    std::string drug_meta_yaml;
    double t_end_days = 365.0;
    double dt_days = 0.1;
};

void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog
        << " --param <xml> [--csv-out <path>] [--binary-out <path>]\n"
        << "                  [--species-out <path>] [--t-end-days N] [--dt-days N]\n"
        << "                  [--scenario <scenario.yaml> --drug-metadata <drug_meta.yaml>]\n"
        << "\n"
        << "Legacy: " << prog
        << " <param_xml> <csv_out> [t_end_days] [dt_days]\n";
}

bool parse_args(int argc, char* argv[], Args& out) {
    int i = 1;
    // Optional legacy positional form: <param_xml> <csv_out> [t_end] [dt].
    // Consume positionals that don't look like flags (argv[k][0] != '-'),
    // then fall through to the flag loop so callers can mix positional
    // with flags like --scenario.
    if (argc > i && argv[i][0] != '-') { out.param_file = argv[i++]; }
    if (argc > i && argv[i][0] != '-') { out.csv_out    = argv[i++]; }
    if (argc > i && argv[i][0] != '-') { out.t_end_days = std::stod(argv[i++]); }
    if (argc > i && argv[i][0] != '-') { out.dt_days    = std::stod(argv[i++]); }

    for (; i < argc; ++i) {
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
        } else if (a == "--scenario") {
            const char* v = need_val("--scenario"); if (!v) return false;
            out.scenario_yaml = v;
        } else if (a == "--drug-metadata") {
            const char* v = need_val("--drug-metadata"); if (!v) return false;
            out.drug_meta_yaml = v;
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
    if (!out.scenario_yaml.empty() && out.drug_meta_yaml.empty()) {
        std::cerr << "--scenario requires --drug-metadata" << std::endl;
        return false;
    }
    return true;
}

// ---- Species-name → index -------------------------------------------
// getHeader() returns "V_C.nCD4,V_C.Treg,...". The Nth name's index in
// _species_var is N. Only a handful of drug-target species are looked up,
// so a linear scan is fine.
int species_index(const std::string& name) {
    static std::vector<std::string> names;
    if (names.empty()) {
        std::string h = ODE_system::getHeader();
        std::string cur;
        for (char c : h) {
            if (c == ',') { names.push_back(cur); cur.clear(); }
            else          { cur += c; }
        }
        if (!cur.empty()) names.push_back(cur);
    }
    for (size_t i = 0; i < names.size(); i++) {
        if (names[i] == name) return static_cast<int>(i);
    }
    return -1;
}

// ---- Dose scheduling ------------------------------------------------
struct Bolus {
    double t_sim;        // absolute simulation time in solver units
    int species_idx;
    double amount;       // in the storage unit of the species (SI moles, cells, etc.)
    std::string label;   // for logging
};

double scale_amount_to_storage(const std::string& units, double amount_in_units) {
    // Storage is SI substance. The wrapper applies doses as "add N mol to the
    // amount" rather than "set concentration", mirroring MATLAB sbiodose with
    // AmountUnits='mole'.
    if (units == "mole")      return amount_in_units;
    if (units == "cell")      return amount_in_units / 6.02214076e23;
    if (units == "milligram") {
        throw std::runtime_error("milligram units not yet supported in dumper");
    }
    throw std::runtime_error("unknown dose units: " + units);
}

// Expand a scenario + drug metadata into a list of Bolus events.
// time_factor converts days to the solver's time unit.
std::vector<Bolus> build_dose_plan(
    const YAML::Node& scenario,
    const YAML::Node& drug_meta,
    double time_factor)
{
    std::vector<Bolus> plan;
    if (!scenario["dosing"]) return plan;
    const auto& dosing = scenario["dosing"];

    const double patient_weight = dosing["patientWeight"]
        ? dosing["patientWeight"].as<double>() : 70.0;
    const double patient_bsa = dosing["patientBSA"]
        ? dosing["patientBSA"].as<double>() : 1.9;

    if (!dosing["drugs"]) return plan;
    for (const auto& drug_n : dosing["drugs"]) {
        std::string drug = drug_n.as<std::string>();

        const auto& meta_drugs = drug_meta["drugs"];
        if (!meta_drugs || !meta_drugs[drug]) {
            throw std::runtime_error("drug not in drug_metadata.yaml: " + drug);
        }
        const auto& md = meta_drugs[drug];
        const std::string units = md["units"].as<std::string>();
        const std::string basis = md["dose_basis"].as<std::string>();

        const std::string dose_key = drug + "_dose";
        const std::string sched_key = drug + "_schedule";
        if (!dosing[dose_key]) {
            throw std::runtime_error("scenario is missing " + dose_key);
        }
        const double raw_dose = dosing[dose_key].as<double>();
        const auto sched = dosing[sched_key].as<std::vector<double>>();
        if (sched.size() != 3) {
            throw std::runtime_error(sched_key + " must be [start, interval, repeat]");
        }
        const double start_day = sched[0];
        const double interval_day = sched[1];
        const int repeat = static_cast<int>(sched[2]);

        double total_amount = 0.0;
        if (basis == "per_weight") {
            const double mw = md["mw"].as<double>();
            total_amount = patient_weight * raw_dose / mw;
        } else if (basis == "per_bsa") {
            const double mw = md["mw"].as<double>();
            total_amount = patient_bsa * raw_dose / mw;
        } else if (basis == "direct") {
            total_amount = raw_dose;
        } else {
            throw std::runtime_error("unknown dose_basis: " + basis);
        }

        for (const auto& target : md["targets"]) {
            const std::string sp_name = target["species"].as<std::string>();
            const double frac = target["fraction"].as<double>();
            const int idx = species_index(sp_name);
            if (idx < 0) {
                throw std::runtime_error("target species not found in ODE: " + sp_name);
            }
            const double storage_amount = scale_amount_to_storage(
                units, total_amount * frac);

            for (int r = 0; r < repeat; r++) {
                const double t_day = start_day + r * interval_day;
                plan.push_back({
                    t_day * time_factor,
                    idx,
                    storage_amount,
                    drug + "@" + sp_name,
                });
            }
        }
    }
    return plan;
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
    double t_end = args.t_end_days * time_factor;
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

    // Load scenario + drug metadata if requested, building the bolus plan
    // before any integration so we know whether to take the fast sampling
    // path or the step-based dosing path.
    std::vector<Bolus> dose_plan;
    if (!args.scenario_yaml.empty()) {
        YAML::Node scenario = YAML::LoadFile(args.scenario_yaml);
        YAML::Node drug_meta = YAML::LoadFile(args.drug_meta_yaml);
        dose_plan = build_dose_plan(scenario, drug_meta, time_factor);
        std::cerr << "Loaded " << dose_plan.size() << " bolus events from "
                  << args.scenario_yaml << std::endl;
        for (const auto& b : dose_plan) {
            std::cerr << "  t=" << (b.t_sim / time_factor) << "d  "
                      << b.label << "  amount=" << b.amount << std::endl;
        }
        if (scenario["sim_config"] && scenario["sim_config"]["stop_time"]) {
            t_end = scenario["sim_config"]["stop_time"].as<double>() * time_factor;
        }
    }

    uint64_t n_times = 0;

    if (dose_plan.empty()) {
        // Fast sampling path: CVODE keeps its internal history across output
        // points (via simOdeSample) so fine output cadence doesn't force
        // the solver to repeatedly restart from a tiny step-size. Event-free.
        ode.setupSamplingRun(t_end);

        write_state(0.0);
        n_times = 1;

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
    } else {
        // Step-based dosing path: boluses are applied between simOdeStep calls
        // via setSpeciesVar(cur + amt). simOdeStep re-inits CVODE each step,
        // which picks up the perturbed state on the next integration chunk.
        // TODO (segmented sampling follow-up): integrate only between dose
        // times with setupSamplingRun, reinit at dose boundaries. See
        // memory/cpp_sim_segmented_sampling_followup.md for design.
        auto apply_due = [&](double t_lo, double t_hi) {
            // Half-open (t_lo, t_hi] for mid-sim; closed [0, 0] for t=0.
            for (const auto& b : dose_plan) {
                bool due = (t_lo == 0.0 && b.t_sim == 0.0) ||
                           (b.t_sim > t_lo && b.t_sim <= t_hi);
                if (due) {
                    double cur = ode.getSpeciesVar(
                        static_cast<unsigned int>(b.species_idx), false);
                    ode.setSpeciesVar(
                        static_cast<unsigned int>(b.species_idx),
                        cur + b.amount, false);
                    std::cerr << "  [dose] t=" << (t_hi / time_factor) << "d  "
                              << b.label << "  +" << b.amount << std::endl;
                }
            }
        };
        apply_due(0.0, 0.0);

        write_state(0.0);
        n_times = 1;

        double t = 0.0;
        int step = 0;
        while (t < t_end) {
            double t_step = dt;
            if (t + t_step > t_end) t_step = t_end - t;
            ode.simOdeStep(t, t_step);
            double t_new = t + t_step;
            apply_due(t, t_new);
            t = t_new;
            step++;
            write_state(t);
            n_times++;

            if (step % 1000 == 0) {
                std::cerr << "  t=" << t / time_factor << " days" << std::endl;
            }
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