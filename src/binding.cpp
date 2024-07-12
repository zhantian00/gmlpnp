#include "ppnp.h"
#include "dlt.h"
#include "gmlpnp_with_prior.h"
#include "gmlpnp.h"
#include "opengv_wrapper.h"
#include "cpnp.h"

// Python binding code
PYBIND11_MODULE(advancedpnp, m) {
    m.doc() = "Python wrapper for pnp.";
    m.def("solve_pnp_ppnp", &solvePnpWithPpnp, "Solve pnp with ppnp.");
    m.def("solve_pnp_dlt", &solvePnPbyDLT, "Solve pnp with DLT.");
    m.def("solve_pnp_mlpnp", &solvePnpByMlpnp, "Opengv wrapper for MLPnP");
    m.def("solve_pnp_upnp", &solvePnpByUpnp, "Opengv wrapper for UPnP");
    m.def("solve_pnp_cpnp", &pnpsolver::CPnP, "wrapper for CPnP");
    m.def("solve_pnp_gmlpnp_with_prior", &solvePnpByGmlpnpWP, "Solve pnp with reconstruction error and scale elimination, known covariance.");
    m.def("solve_pnp_gmlpnp", &solvePnpByGmlpnp, "Solve pnp with reconstruction error and scale elimination and with unknown covariance.");
}