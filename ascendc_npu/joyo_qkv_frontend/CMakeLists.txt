# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0

option(JOYO_QKV_FRONTEND_USE_PREBUILT
  "Use the Joyo QKV Cube/Vector library for Ascend910_9382" ON)
set(JOYO_QKV_FRONTEND_CATLASS_INCLUDE_DIR "" CACHE PATH
  "Catlass include directory used when rebuilding the Joyo QKV kernel")

if(JOYO_QKV_FRONTEND_USE_PREBUILT)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/manifest.json" joyo_qkv_manifest)
  string(JSON joyo_qkv_relative_library GET "${joyo_qkv_manifest}" library)
  string(JSON joyo_qkv_expected_sha GET "${joyo_qkv_manifest}" sha256)
  set(joyo_qkv_library
    "${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/${joyo_qkv_relative_library}")
  file(SHA256 "${joyo_qkv_library}" joyo_qkv_actual_sha)
  if(NOT joyo_qkv_actual_sha STREQUAL joyo_qkv_expected_sha)
    message(FATAL_ERROR "Joyo QKV library checksum mismatch")
  endif()
  add_library(joyo_qkv_frontend SHARED IMPORTED GLOBAL)
  set_target_properties(joyo_qkv_frontend PROPERTIES
    IMPORTED_LOCATION "${joyo_qkv_library}")
else()
  if(NOT EXISTS "${JOYO_QKV_FRONTEND_CATLASS_INCLUDE_DIR}/catlass/catlass.hpp")
    message(FATAL_ERROR "Set JOYO_QKV_FRONTEND_CATLASS_INCLUDE_DIR to Catlass include/")
  endif()
  set(joyo_qkv_cann_root "$ENV{ASCEND_HOME_PATH}")
  if(NOT joyo_qkv_cann_root)
    set(joyo_qkv_cann_root "/usr/local/Ascend/ascend-toolkit/latest")
  endif()
  find_program(joyo_qkv_bisheng bisheng
    HINTS "${joyo_qkv_cann_root}/compiler/ccec_compiler/bin" REQUIRED)
  set(joyo_qkv_object "${CMAKE_CURRENT_BINARY_DIR}/joyo_qkv_frontend.o")
  set(joyo_qkv_library "${CMAKE_CURRENT_BINARY_DIR}/libjoyo_qkv_frontend.so")
  add_custom_command(OUTPUT "${joyo_qkv_object}"
    COMMAND "${joyo_qkv_bisheng}"
      "-I${CMAKE_CURRENT_SOURCE_DIR}/private_include"
      -x asc --npu-arch=dav-2201 -DCATLASS_ARCH=2201 -O2 -std=c++17 -fPIC
      -c "${CMAKE_CURRENT_SOURCE_DIR}/joyo_qkv_frontend.asc"
      "-I${JOYO_QKV_FRONTEND_CATLASS_INCLUDE_DIR}"
      "-I${joyo_qkv_cann_root}/aarch64-linux/asc/include"
      "-I${joyo_qkv_cann_root}/aarch64-linux/asc/include/adv_api"
      "-I${joyo_qkv_cann_root}/aarch64-linux/asc/include/basic_api"
      "-I${joyo_qkv_cann_root}/aarch64-linux/asc/include/interface"
      "-I${joyo_qkv_cann_root}/aarch64-linux/include"
      -o "${joyo_qkv_object}"
    DEPENDS joyo_qkv_frontend.asc
      private_include/catlass/gemm/block/block_mmad_pingpong.hpp
    VERBATIM)
  add_custom_command(OUTPUT "${joyo_qkv_library}"
    COMMAND "${CMAKE_CXX_COMPILER}" -shared -fPIC "${joyo_qkv_object}"
      "-L${joyo_qkv_cann_root}/aarch64-linux/lib64"
      "-Wl,-rpath,${joyo_qkv_cann_root}/aarch64-linux/lib64"
      -lascendc_runtime -lprofapi -lmmpa -lunified_dlog -lc_sec -ltiling_api
      -lplatform -lregister -lascendcl -lruntime -ldl -o "${joyo_qkv_library}"
    DEPENDS "${joyo_qkv_object}" VERBATIM)
  add_custom_target(joyo_qkv_frontend_build DEPENDS "${joyo_qkv_library}")
  add_library(joyo_qkv_frontend SHARED IMPORTED GLOBAL)
  set_target_properties(joyo_qkv_frontend PROPERTIES
    IMPORTED_LOCATION "${joyo_qkv_library}")
  add_dependencies(joyo_qkv_frontend joyo_qkv_frontend_build)
endif()

add_dependencies(torch_npu_kernels joyo_qkv_frontend)
target_compile_definitions(torch_npu_kernels PUBLIC
  "XLLM_JOYO_QKV_FRONTEND_DEFAULT_LIBRARY=\"${joyo_qkv_library}\"")
