#pragma once

enum LookupMode {
  LOOKUP_MODE_1D = 0,
  LOOKUP_MODE_3D = 1,
  LOOKUP_MODE_FULL = 2,
  LOOKUP_MODE_HYBRID = 3,
  LOOKUP_MODE_SMART = 4
};

extern LookupMode g_lookup_mode;
extern bool g_print_runtime_config;
