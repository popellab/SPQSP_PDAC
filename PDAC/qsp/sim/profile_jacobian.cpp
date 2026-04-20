/**
 * Profile the QSP ODE system's Jacobian sparsity at t=0 on the template IC.
 *
 * Method: finite-difference column-by-column. For each species i, perturb
 * y[i] by max(|y[i]| * rel_eps, abs_eps), call f, subtract baseline, divide.
 * Count nnz under a tolerance and report density + per-row fan-in stats.
 *
 * This tells us whether "analytical Jacobian + KLU" is worth chasing:
 *   - <5% dense: very sparse, KLU likely 2-5× speedup (matches doc prediction)
 *   - 5-15%:     moderately sparse, KLU probably 1.5-2×
 *   - >15%:      fill-in eats most of the win, KLU marginal
 *
 * Usage: ./profile_jacobian <param_xml> [t_eval_days]
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include "qsp/ode/ODE_system.h"
#include "qsp/ode/QSPParam.h"

using namespace CancerVCT;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <param_xml>\n";
        return 1;
    }

    QSPParam param;
    param.initializeParams(argv[1]);
    ODE_system::setup_class_parameters(param);

    ODE_system ode;
    ode.setup_instance_variables(param);
    ode.setup_instance_tolerance(param);
    ode.eval_init_assignment();

    const double t_eval_days = argc > 2 ? std::stod(argv[2]) : 0.0;
#ifdef MODEL_UNITS
    const double time_factor = 1.0;
#else
    const double time_factor = 86400.0;
#endif
    const double t_eval = t_eval_days * time_factor;

    // setupSamplingRun pushes _species_var into _y. If we want the state
    // at t>0, advance the solver to that time first.
    ode.setupSamplingRun(std::max(t_eval, 1.0));
    if (t_eval > 0.0) ode.simOdeSample(t_eval);

    // Access _y via the CVODEBase handle indirectly: grab the state by
    // evaluating getSpeciesOutputValue at each species index... actually
    // simpler: we allocate our own N_Vector, fill from the current state.
    const int n = ode.getNumOutputSpecies();
    std::vector<double> y(n), y_pert(n), f0(n), f1(n);
    for (int i = 0; i < n; ++i) {
        // getSpeciesOutputValue returns original units; we need the raw
        // solver state. Use getSpeciesVar with raw=true.
        // But getSpeciesVar uses unsigned int idx and handles _neq only.
        y[i] = (i < param.getVal(0) + 1000)  // bounds hint
                   ? ode.getSpeciesVar(static_cast<unsigned int>(i), true)
                   : 0.0;
    }
    // neq is the ODE subset — only those columns participate in J.
    // We can't directly read _neq from outside, so peel it from the
    // sum: number_output - number_other. We don't know number_other
    // either. Workaround: call f once; the returned ydot length equals
    // _neq (the CVODE state dimension). Use an N_Vector matched to
    // getNumOutputSpecies() — works because f reads only y[0.._neq-1].

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);
    N_Vector Y  = N_VNew_Serial(n, ctx);
    N_Vector Yp = N_VNew_Serial(n, ctx);
    N_Vector F0 = N_VNew_Serial(n, ctx);
    N_Vector F1 = N_VNew_Serial(n, ctx);

    for (int i = 0; i < n; ++i) NV_DATA_S(Y)[i] = y[i];

    // Baseline f(0, y).
    ODE_system::f(0.0, Y, F0, static_cast<void*>(&ode));

    // Figure out the real neq by observing which rows of F0 are nontrivial
    // structurally. Simpler: assume neq == n for now; zeros are fine
    // (derivatives of pure assignment-rule species w.r.t. y entries are
    // zero, so those rows are empty — they don't inflate density).
    const int neq = n;

    const double rel_eps = 1e-7;
    const double abs_eps = 1e-12;
    const double tol = 1e-8;  // nnz threshold, scale-normalized

    // J[i][j] = ∂f_i/∂y_j (row i, column j). Column-major perturbation.
    std::vector<std::vector<double>> J(neq, std::vector<double>(n, 0.0));

    for (int j = 0; j < n; ++j) {
        for (int k = 0; k < n; ++k) NV_DATA_S(Yp)[k] = y[k];
        double eps = std::max(std::fabs(y[j]) * rel_eps, abs_eps);
        NV_DATA_S(Yp)[j] += eps;
        ODE_system::f(0.0, Yp, F1, static_cast<void*>(&ode));
        for (int i = 0; i < neq; ++i) {
            double df = NV_DATA_S(F1)[i] - NV_DATA_S(F0)[i];
            J[i][j] = df / eps;
        }
    }

    // Stats.
    // Use scale-normalized tolerance: J_ij is "nonzero" if
    // |J_ij| > tol * max(|f_i|/y_scale, row_scale)
    // Simpler: absolute threshold on normalized columns.
    std::vector<double> row_max(neq, 0.0);
    for (int i = 0; i < neq; ++i) {
        for (int j = 0; j < n; ++j) {
            row_max[i] = std::max(row_max[i], std::fabs(J[i][j]));
        }
    }

    long long nnz = 0, n_total = static_cast<long long>(neq) * n;
    std::vector<int> row_nnz(neq, 0);
    for (int i = 0; i < neq; ++i) {
        if (row_max[i] == 0.0) continue;  // all-zero row — truly empty
        for (int j = 0; j < n; ++j) {
            if (std::fabs(J[i][j]) > tol * row_max[i]) {
                ++nnz;
                ++row_nnz[i];
            }
        }
    }

    int empty_rows = 0, diag_only = 0;
    for (int i = 0; i < neq; ++i) {
        if (row_nnz[i] == 0) empty_rows++;
        else if (row_nnz[i] == 1) diag_only++;
    }

    std::sort(row_nnz.begin(), row_nnz.end());
    int max_row = row_nnz.back();
    int median_row = row_nnz[neq / 2];

    std::printf("=== Jacobian sparsity @ t=%.1f days ===\n", t_eval_days);
    std::printf("  size:           %d × %d (%lld entries)\n", neq, n, n_total);
    std::printf("  nnz:            %lld\n", nnz);
    std::printf("  density:        %.2f%% (%.4f)\n",
                100.0 * nnz / n_total, static_cast<double>(nnz) / n_total);
    std::printf("  empty rows:     %d\n", empty_rows);
    std::printf("  singleton rows: %d\n", diag_only);
    std::printf("  max row nnz:    %d  (fan-in for most-connected species)\n", max_row);
    std::printf("  median row nnz: %d\n", median_row);

    // Histogram
    std::printf("\n  row-nnz histogram (bucketed):\n");
    int buckets[7] = {0};
    for (int i = 0; i < neq; ++i) {
        int r = row_nnz[i];
        int b = r == 0 ? 0 : r <= 2 ? 1 : r <= 5 ? 2 : r <= 10 ? 3 : r <= 20 ? 4 : r <= 50 ? 5 : 6;
        buckets[b]++;
    }
    const char* labels[] = {"0", "1-2", "3-5", "6-10", "11-20", "21-50", "51+"};
    for (int b = 0; b < 7; ++b) {
        std::printf("    %-6s  %4d rows  %s\n", labels[b], buckets[b],
                    std::string(buckets[b] / 2, '#').c_str());
    }

    N_VDestroy(Y); N_VDestroy(Yp); N_VDestroy(F0); N_VDestroy(F1);
    SUNContext_Free(&ctx);
    return 0;
}
