cmake_minimum_required(VERSION 3.4.1)

project(ItcMultiDetect)

#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/ItcMultiDetect_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

set(LIB_ARCH aarch64)
if (CMAKE_C_COMPILER MATCHES "aarch64")
  set(LIB_ARCH aarch64)
else()
  set(LIB_ARCH armhf)
endif()
include_directories(${CMAKE_SOURCE_DIR})

# rknn api
if(TARGET_SOC STREQUAL "rk356x")
  set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../runtime/RK356X/${CMAKE_SYSTEM_NAME}/librknn_api)
elseif(TARGET_SOC STREQUAL "rk3588")
  set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../runtime/RK3588/${CMAKE_SYSTEM_NAME}/librknn_api)
else()
  message(FATAL_ERROR "TARGET_SOC is not set, ref value: rk356x or rk3588 or rv110x")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${CMAKE_ANDROID_ARCH_ABI}/librknnrt.so)
else()
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
endif()
include_directories(${RKNN_API_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/../3rdparty)

# opencv
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../3rdparty/opencv/OpenCV-android-sdk/sdk/native/jni/abi-${CMAKE_ANDROID_ARCH_ABI})
else()
  if(LIB_ARCH STREQUAL "armhf")
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../3rdparty/opencv/opencv-linux-armhf/share/OpenCV)
  else()
    if (GCC_VERSION STREQUAL "7.5")
      set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../3rdparty/GCC_7_5/opencv/opencv-linux-aarch64/share/OpenCV)
    else()
      set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../3rdparty/GCC_10_4/opencv/opencv-linux-aarch64/share/OpenCV)
    endif()
  endif()
endif()
find_package(OpenCV REQUIRED)


set(CMAKE_INSTALL_RPATH "lib")

include_directories( ${CMAKE_SOURCE_DIR}/include)

include_directories( ${CMAKE_SOURCE_DIR}/src)

add_library(ItcMultiDetect SHARED 
            src/ItcMultiDetect.cc 
            src/resize_function.cc
            src/rknn_utils.cc
            src/postprocess.cc
            src/ItcMultiDetectImpl.cc         
            )

target_link_libraries(ItcMultiDetect
  ${RKNN_RT_LIB}
  ${OpenCV_LIBS}
)

#链接phtead库
find_package(Threads REQUIRED)
target_link_libraries(ItcMultiDetect ${CMAKE_THREAD_LIBS_INIT})

