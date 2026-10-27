# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0

set(JOYO_LASER_VENDOR_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/ascend910_93/aie_ascendc"
  CACHE PATH "LaserAttention CANN vendor directory; source its bin/set_env.bash before serving")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/manifest.json" joyo_laser_manifest)
string(JSON joyo_laser_artifacts GET "${joyo_laser_manifest}" artifacts)
string(JSON joyo_laser_count LENGTH "${joyo_laser_artifacts}")
math(EXPR joyo_laser_last "${joyo_laser_count} - 1")
foreach(index RANGE 0 ${joyo_laser_last})
  string(JSON relative MEMBER "${joyo_laser_artifacts}" ${index})
  string(JSON expected GET "${joyo_laser_artifacts}" "${relative}")
  file(SHA256 "${JOYO_LASER_VENDOR_PATH}/${relative}" actual)
  if(NOT actual STREQUAL expected)
    message(FATAL_ERROR "LaserAttention vendor checksum mismatch: ${relative}")
  endif()
endforeach()
add_library(joyo_laser_opapi SHARED IMPORTED GLOBAL)
set_target_properties(joyo_laser_opapi PROPERTIES
  IMPORTED_LOCATION "${JOYO_LASER_VENDOR_PATH}/op_api/lib/libcust_opapi.so"
  INTERFACE_INCLUDE_DIRECTORIES "${JOYO_LASER_VENDOR_PATH}/op_api/include")
# CANN discovers this optional vendor through ASCEND_CUSTOM_OPP_PATH.
# Keep FA's dependency loading unchanged when Laser is not selected.
install(DIRECTORY "${JOYO_LASER_VENDOR_PATH}"
  DESTINATION share/xllm/vendors)
message(STATUS "Joyo LaserAttention vendor: ${JOYO_LASER_VENDOR_PATH}")
