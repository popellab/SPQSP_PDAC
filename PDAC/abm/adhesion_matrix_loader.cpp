#include "adhesion_matrix_loader.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <iostream>
#include <unordered_map>
#include <stdexcept>

namespace PDAC {

namespace pt = boost::property_tree;

// XML tag → StateCounterIdx integer value. Must stay in sync with
// `enum StateCounterIdx` in PDAC/core/common.cuh and SC_TAG_MAP in
// PDAC/codegen/abm_param_codegen.py.
static const std::unordered_map<std::string, int>& sc_tag_map() {
    static const std::unordered_map<std::string, int> m = {
        {"cancer_stem",       0},
        {"cancer_prog",       1},
        {"cancer_sen",        2},
        {"cd8_eff",           3},
        {"cd8_cyt",           4},
        {"cd8_sup",           5},
        {"cd8_naive",         6},
        {"th",                7},
        {"treg",              8},
        {"tfh",               9},
        {"tcd4_naive",       10},
        {"mdsc",             11},
        {"mac_m1",           12},
        {"mac_m2",           13},
        {"fib_quiescent",    14},
        {"fib_mycaf",        15},
        {"fib_icaf",         16},
        {"fib_frc",          17},
        {"vas_tip",          18},
        {"vas_phalanx",      19},
        {"vas_collapsed",    20},
        {"vas_hev",          21},
        {"bcell_naive",      22},
        {"bcell_act",        23},
        {"bcell_plasma",     24},
        {"dc_cdc1_immature", 25},
        {"dc_cdc1_mature",   26},
        {"dc_cdc2_immature", 27},
        {"dc_cdc2_mature",   28},
    };
    return m;
}

void load_adhesion_matrix_from_xml(const std::string& xml_path,
                                   float* h_matrix,
                                   int N) {
    pt::ptree tree;
    try {
        pt::read_xml(xml_path, tree, pt::xml_parser::trim_whitespace);
    } catch (const std::exception& e) {
        std::cerr << "load_adhesion_matrix_from_xml: failed to read "
                  << xml_path << ": " << e.what() << std::endl;
        std::exit(1);
    }

    // Walk Param.ABM.Movement.AdhesionMatrix; if missing, leave matrix at zero.
    auto am_opt = tree.get_child_optional("Param.ABM.Movement.AdhesionMatrix");
    if (!am_opt) {
        std::cout << "Adhesion matrix: <Param>/<ABM>/<Movement>/<AdhesionMatrix> not found in "
                  << xml_path << " — matrix left at zero." << std::endl;
        return;
    }

    const auto& tag_map = sc_tag_map();
    int n_entries = 0;
    int n_warnings = 0;

    for (const auto& row_kv : *am_opt) {
        const std::string& row_tag = row_kv.first;
        if (row_tag == "<xmlattr>" || row_tag == "<xmlcomment>") continue;

        auto row_it = tag_map.find(row_tag);
        if (row_it == tag_map.end()) {
            std::cerr << "load_adhesion_matrix_from_xml: unknown row tag '"
                      << row_tag << "' — skipping." << std::endl;
            ++n_warnings;
            continue;
        }
        int row_sc = row_it->second;
        if (row_sc < 0 || row_sc >= N) {
            std::cerr << "load_adhesion_matrix_from_xml: row index "
                      << row_sc << " (tag '" << row_tag << "') out of range [0,"
                      << N << ") — skipping." << std::endl;
            ++n_warnings;
            continue;
        }

        for (const auto& col_kv : row_kv.second) {
            const std::string& col_tag = col_kv.first;
            if (col_tag == "<xmlattr>" || col_tag == "<xmlcomment>") continue;

            auto col_it = tag_map.find(col_tag);
            if (col_it == tag_map.end()) {
                std::cerr << "load_adhesion_matrix_from_xml: unknown col tag '"
                          << col_tag << "' under row '" << row_tag
                          << "' — skipping." << std::endl;
                ++n_warnings;
                continue;
            }
            int col_sc = col_it->second;
            if (col_sc < 0 || col_sc >= N) {
                std::cerr << "load_adhesion_matrix_from_xml: col index "
                          << col_sc << " (tag '" << col_tag << "') out of range [0,"
                          << N << ") — skipping." << std::endl;
                ++n_warnings;
                continue;
            }

            float val = 0.0f;
            try {
                val = col_kv.second.get_value<float>();
            } catch (const std::exception& e) {
                std::cerr << "load_adhesion_matrix_from_xml: bad float at "
                          << row_tag << "/" << col_tag << ": " << e.what()
                          << std::endl;
                ++n_warnings;
                continue;
            }

            h_matrix[row_sc * N + col_sc] = val;
            ++n_entries;
        }
    }

    std::cout << "Adhesion matrix: loaded " << n_entries
              << " non-zero entries from " << xml_path;
    if (n_warnings) std::cout << " (" << n_warnings << " warnings)";
    std::cout << std::endl;
}

}  // namespace PDAC
