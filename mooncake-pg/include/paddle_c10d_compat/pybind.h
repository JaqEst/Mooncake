#pragma once

#include <pybind11/pybind11.h>

namespace mooncake::paddle_c10d_compat {

void bindC10dCompat(pybind11::module_& m);

} // namespace mooncake::paddle_c10d_compat
