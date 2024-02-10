function(_execute_process_checked)
  cmake_parse_arguments(EXECUTE_PROCESS_CHECKED "" "" "COMMAND" ${ARGN})
  execute_process(COMMAND ${EXECUTE_PROCESS_CHECKED_COMMAND} RESULT_VARIABLE result)
  if(NOT "${result}" EQUAL "0")
    message(FATAL_ERROR "`${EXECUTE_PROCESS_CHECKED_COMMAND}` failed")
  endif()
endfunction()

include(FetchContent)

FetchContent_Declare(
  Corrosion
  GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
  GIT_TAG v0.5.0 # We do a hack for iOS.
)
FetchContent_MakeAvailable(Corrosion)

if(IOS)
  if("${Rust_CARGO_TARGET}" STREQUAL "")
    message(FATAL_ERROR "`$Rust_CARGO_TARGET` must be set")
  endif()
  add_library(decrypt_vv_model STATIC IMPORTED)
  # Runs `cargo build` now, not in build time. Running `cargo build` in build time somehow causes
  # https://github.com/corrosion-rs/corrosion/issues/104
  _execute_process_checked(
    COMMAND
    cargo build
    --manifest-path ../../decrypt_vv_model/Cargo.toml
    --target "${Rust_CARGO_TARGET}"
    --release
  )
  set_target_properties(
    decrypt_vv_model
    PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../decrypt_vv_model/target/${Rust_CARGO_TARGET}/release/libdecrypt_vv_model.a"
    # for `corrosion_add_cxxbridge`, set a property as `corrosion_import_crate` does
    INTERFACE_COR_PACKAGE_MANIFEST_PATH ../../decrypt_vv_model/Cargo.toml
  )
else()
  corrosion_import_crate(MANIFEST_PATH ../decrypt_vv_model/Cargo.toml)
endif()

corrosion_add_cxxbridge(decrypt_vv_model_cxx CRATE decrypt_vv_model FILES lib.rs)

list(APPEND onnxruntime_EXTERNAL_LIBRARIES decrypt_vv_model_cxx decrypt_vv_model)
list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES decrypt_vv_model_cxx)
