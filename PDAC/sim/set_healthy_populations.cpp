#include "set_healthy_populations.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "qsp/ode/ODE_system.h"
#include "qsp/ode/QSP_enum.h"

namespace CancerVCT {
namespace {

// Resolve a species name (as emitted by ODE_system::getHeader) to its
// index in _species_var. Linear scan is fine; cached after first call.
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

// Try two names: fully-qualified "V_T.Foo" and bare "Foo". Return the
// first hit, or -1 if neither exists.
int find_v_t_species(const std::string& bare) {
    int idx = species_index("V_T." + bare);
    if (idx >= 0) return idx;
    return species_index(bare);
}

void set_cell_count(ODE_system& ode, const std::string& bare, double N_cells,
                    bool verbose) {
    int idx = find_v_t_species(bare);
    if (idx < 0) {
        if (verbose) std::cerr << "  [healthy] " << bare
                                << " — species not found, skipping\n";
        return;
    }
    // Cell-unit species: storage factor is 1/NA (mol/cell). setSpeciesVar
    // with raw=true multiplies by the factor, so pass N_cells directly.
    ode.setSpeciesVar(idx, N_cells, true);
    if (verbose) {
        std::cerr << "  [healthy] " << bare << " = " << N_cells << " cells\n";
    }
}

}  // namespace

void set_healthy_populations(ODE_system& ode, const HealthyPopulationOpts& opts) {
    if (opts.yaml_path.empty()) {
        throw std::runtime_error("set_healthy_populations: yaml_path is required");
    }
    YAML::Node y = YAML::LoadFile(opts.yaml_path);

    const double tumor_cells = (opts.tumor_cells > 0.0)
        ? opts.tumor_cells
        : y["default_tumor_cells"].as<double>();

    // D_cell is stored in SI meters (*1e-6 from µm in param_all.xml).
    const double D_cell_m = ODE_system::get_class_param(P_D_cell);
    const double D_cell_um = D_cell_m * 1e6;
    const double vol_cell_um3 = (4.0 / 3.0) * M_PI
        * std::pow(D_cell_um * 0.5, 3.0);
    const double vol_cell_mL = vol_cell_um3 * 1e-12;  // 1 mL = 1e12 µm³
    const double V_tumor_mL = tumor_cells * vol_cell_mL;
    const double V_tumor_mm3 = V_tumor_mL * 1e3;      // 1 mL = 1e3 mm³

    if (opts.verbose) {
        std::cerr << "[healthy] D_cell=" << D_cell_um << " um; vol_cell="
                  << vol_cell_um3 << " um^3; V_tumor=" << V_tumor_mm3
                  << " mm^3 (" << V_tumor_mL << " mL)\n";
    }

    // --- Cell densities → absolute counts --------------------------------
    const auto& dens = y["cell_densities_per_mm3"];
    const auto& rat  = y["ratios"];
    const double m1_m2_ratio = rat["m1_m2"].as<double>();
    const double f_cDC1      = rat["f_cDC1"].as<double>();
    const double apc_mat     = rat["apc_mat"].as<double>();

    const double CD8_N    = dens["CD8"].as<double>()   * V_tumor_mm3;
    const double Treg_N   = dens["Treg"].as<double>()  * V_tumor_mm3;
    const double Th_N     = dens["Th"].as<double>()    * V_tumor_mm3;
    const double MDSC_N   = dens["MDSC"].as<double>()  * V_tumor_mm3;
    const double iCAF_N   = dens["iCAF"].as<double>()  * V_tumor_mm3;
    const double myCAF_N  = dens["myCAF"].as<double>() * V_tumor_mm3;
    const double apCAF_N  = dens["apCAF"].as<double>() * V_tumor_mm3;
    const double qPSC_N   = dens["qPSC"].as<double>()  * V_tumor_mm3;

    const double TAM_total = dens["TAM_total"].as<double>() * V_tumor_mm3;
    const double Mac_M2_N  = TAM_total / (1.0 + m1_m2_ratio);
    const double Mac_M1_N  = m1_m2_ratio * Mac_M2_N;

    const double APC_total = dens["APC_total"].as<double>() * V_tumor_mm3;
    const double cDC1_N    = f_cDC1 * (1.0 - apc_mat) * APC_total;
    const double cDC2_N    = (1.0 - f_cDC1) * (1.0 - apc_mat) * APC_total;
    const double mcDC1_N   = f_cDC1 * apc_mat * APC_total;
    const double mcDC2_N   = (1.0 - f_cDC1) * apc_mat * APC_total;

    set_cell_count(ode, "C1",    tumor_cells, opts.verbose);
    set_cell_count(ode, "K",     tumor_cells, opts.verbose);
    set_cell_count(ode, "CD8",   CD8_N,   opts.verbose);
    set_cell_count(ode, "Treg",  Treg_N,  opts.verbose);
    set_cell_count(ode, "Th",    Th_N,    opts.verbose);
    set_cell_count(ode, "MDSC",  MDSC_N,  opts.verbose);
    set_cell_count(ode, "Mac_M1", Mac_M1_N, opts.verbose);
    set_cell_count(ode, "Mac_M2", Mac_M2_N, opts.verbose);
    set_cell_count(ode, "iCAF",  iCAF_N,  opts.verbose);
    set_cell_count(ode, "myCAF", myCAF_N, opts.verbose);
    set_cell_count(ode, "apCAF", apCAF_N, opts.verbose);
    set_cell_count(ode, "qPSC",  qPSC_N,  opts.verbose);
    set_cell_count(ode, "cDC1",  cDC1_N,  opts.verbose);
    set_cell_count(ode, "cDC2",  cDC2_N,  opts.verbose);
    set_cell_count(ode, "mcDC1", mcDC1_N, opts.verbose);
    set_cell_count(ode, "mcDC2", mcDC2_N, opts.verbose);

    // --- Collagen (mass in mg → storage kg via factor 1e-6) --------------
    const auto& col = y["collagen"];
    const double col_frac = col["volume_fraction"].as<double>();
    const double col_dens = col["density_mg_per_mL"].as<double>();
    const double collagen_mg = col_frac * V_tumor_mL * col_dens;
    int col_idx = find_v_t_species("collagen");
    if (col_idx >= 0) {
        ode.setSpeciesVar(col_idx, collagen_mg, true);
        if (opts.verbose) {
            std::cerr << "  [healthy] collagen = " << collagen_mg << " mg\n";
        }
    }

    // --- Cytokines -------------------------------------------------------
    // Storage for concentration species in V_T is conc × V_T × scale where
    // scale matches setup_instance_variables (pg/mL → 1e-15; nM → 1e-12).
    // setSpeciesVar(raw=true) multiplies by the stored unit factor, so pass
    // (conc × V_T_mL) and rely on the factor built into the species.
    const double V_T_mL = ode.get_compartment_volume("V_T");
    const auto& cyto = y["basal_cytokines"];
    const double CCL2_nM = cyto["CCL2_nM"].as<double>();
    const double TGFb_nM = cyto["TGFb_nM"].as<double>();
    const double VEGF_pg = cyto["VEGF_pg_per_mL"].as<double>();

    auto set_conc = [&](const std::string& bare, double conc_times_vol,
                        const char* unit) {
        int idx = find_v_t_species(bare);
        if (idx < 0) {
            if (opts.verbose) std::cerr << "  [healthy] " << bare
                                         << " — species not found\n";
            return;
        }
        ode.setSpeciesVar(idx, conc_times_vol, true);
        if (opts.verbose) {
            std::cerr << "  [healthy] " << bare << " = " << conc_times_vol
                      << " (" << unit << "·mL)\n";
        }
    };
    set_conc("CCL2", CCL2_nM * V_T_mL, "nM");
    set_conc("TGFb", TGFb_nM * V_T_mL, "nM");
    set_conc("VEGF", VEGF_pg * V_T_mL, "pg/mL");

    // --- Zero out treatment / dynamic species ----------------------------
    for (const auto& n : y["zero_species"]) {
        const std::string bare = n.as<std::string>();
        int idx = find_v_t_species(bare);
        if (idx >= 0) {
            ode.setSpeciesVar(idx, 0.0, true);
            if (opts.verbose) std::cerr << "  [healthy] zero " << bare << "\n";
        }
    }

    // Re-run initial-assignment rules so any derived quantities agree with
    // the new IC before integration resumes.
    ode.eval_init_assignment();
}

}  // namespace CancerVCT