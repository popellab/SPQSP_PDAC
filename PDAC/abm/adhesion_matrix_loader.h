#ifndef ADHESION_MATRIX_LOADER_H
#define ADHESION_MATRIX_LOADER_H

#include <string>

namespace PDAC {

// Parse <Param>/<Movement>/<AdhesionMatrix> from xml_path and fill h_matrix
// (N x N float, row-major). Unknown row/col tags warn and skip. Missing block
// leaves matrix at 0. Aborts on file/parse errors (matching ParamBase).
//
// h_matrix must be pre-zeroed by the caller; loader only writes non-zero
// entries listed in the XML.
void load_adhesion_matrix_from_xml(const std::string& xml_path,
                                   float* h_matrix,
                                   int N);

}  // namespace PDAC

#endif  // ADHESION_MATRIX_LOADER_H
