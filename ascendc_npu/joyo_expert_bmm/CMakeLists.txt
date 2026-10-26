# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0

option(JOYO_EXPERT_BMM_USE_PREBUILT
  "Use the Joyo expert direct NZ BMM library for Ascend910_9382" ON)
set(JOYO_EXPERT_BMM_CATLASS_INCLUDE_DIR "" CACHE PATH
  "Catlass include directory used when rebuilding the Joyo expert NZ BMM kernel")

if(JOYO_EXPERT_BMM_USE_PREBUILT)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/manifest.json" joyo_bmm_manifest)
  string(JSON joyo_bmm_relative_library GET "${joyo_bmm_manifest}" library)
  string(JSON joyo_bmm_expected_sha GET "${joyo_bmm_manifest}" sha256)
  set(joyo_bmm_library
    "${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/${joyo_bmm_relative_library}")
  file(SHA256 "${joyo_bmm_library}" joyo_bmm_actual_sha)
  if(NOT joyo_bmm_actual_sha STREQUAL joyo_bmm_expected_sha)
    message(FATAL_ERROR "Joyo expert NZ BMM library checksum mismatch")
  endif()
  add_library(native_order_bmm SHARED IMPORTED GLOBAL)
  set_target_properties(native_order_bmm PROPERTIES
    IMPORTED_LOCATION "${joyo_bmm_library}")
else()
  if(NOT EXISTS "${JOYO_EXPERT_BMM_CATLASS_INCLUDE_DIR}/catlass/catlass.hpp")
    message(FATAL_ERROR "Set JOYO_EXPERT_BMM_CATLASS_INCLUDE_DIR to Catlass include/")
  endif()
  set(joyo_bmm_cann_root "$ENV{ASCEND_HOME_PATH}")
  if(NOT joyo_bmm_cann_root)
    set(joyo_bmm_cann_root "/usr/local/Ascend/ascend-toolkit/latest")
  endif()
  find_program(joyo_bmm_bisheng bisheng
    HINTS "${joyo_bmm_cann_root}/compiler/ccec_compiler/bin" REQUIRED)
  set(joyo_bmm_object "${CMAKE_CURRENT_BINARY_DIR}/native_order_bmm.o")
  set(joyo_bmm_library "${CMAKE_CURRENT_BINARY_DIR}/libnative_order_bmm.so")
  add_custom_command(OUTPUT "${joyo_bmm_object}"
    COMMAND "${joyo_bmm_bisheng}"
      "-I${CMAKE_CURRENT_SOURCE_DIR}/private_include"
      -x asc --npu-arch=dav-2201 -DCATLASS_ARCH=2201 -O2 -std=c++17 -fPIC
      -c "${CMAKE_CURRENT_SOURCE_DIR}/native_order_bmm.asc"
      "-I${JOYO_EXPERT_BMM_CATLASS_INCLUDE_DIR}"
      "-I${joyo_bmm_cann_root}/aarch64-linux/asc/include"
      "-I${joyo_bmm_cann_root}/aarch64-linux/asc/include/adv_api"
      "-I${joyo_bmm_cann_root}/aarch64-linux/asc/include/basic_api"
      "-I${joyo_bmm_cann_root}/aarch64-linux/asc/include/interface"
      "-I${joyo_bmm_cann_root}/aarch64-linux/include"
      -o "${joyo_bmm_object}"
    DEPENDS native_order_bmm.asc
      private_include/catlass/gemm/block/block_mmad_pingpong.hpp
    VERBATIM)
  add_custom_command(OUTPUT "${joyo_bmm_library}"
    COMMAND "${CMAKE_CXX_COMPILER}" -shared -fPIC "${joyo_bmm_object}"
      "-L${joyo_bmm_cann_root}/aarch64-linux/lib64"
      "-Wl,-rpath,${joyo_bmm_cann_root}/aarch64-linux/lib64"
      -lascendc_runtime -lprofapi -lmmpa -lunified_dlog -lc_sec -ltiling_api
      -lplatform -lregister -lascendcl -lruntime -ldl -o "${joyo_bmm_library}"
    DEPENDS "${joyo_bmm_object}" VERBATIM)
  add_custom_target(native_order_bmm_build DEPENDS "${joyo_bmm_library}")
  add_library(native_order_bmm SHARED IMPORTED GLOBAL)
  set_target_properties(native_order_bmm PROPERTIES
    IMPORTED_LOCATION "${joyo_bmm_library}")
  add_dependencies(native_order_bmm native_order_bmm_build)
endif()

add_dependencies(torch_npu_kernels native_order_bmm)
target_compile_definitions(torch_npu_kernels PUBLIC
  "XLLM_JOYO_EXPERT_NZ_DEFAULT_LIBRARY=\"${joyo_bmm_library}\"")
