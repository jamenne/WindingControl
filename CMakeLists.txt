cmake_minimum_required(VERSION 2.8)

project(WindingControl)
set(PROJECT_VERSION 0.0.0.0)
set(PROJECT_BRIEF "\"No description\"")

set(CMAKE_CXX_FLAGS_DBG "-O0 -ggdb" CACHE STRING "Debug options." FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "Debug options." FORCE)
SET(CMAKE_CXX_FLAGS_PROFILING "-O3 -pg" CACHE STRING "Debug options." FORCE)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")

# ROOT, a library for HEP data analysis (contains RooFit)
find_package(ROOT REQUIRED)
include_directories(${ROOT_INCLUDE_DIR})
set(LIBRARIES ${LIBRARIES} ${ROOT_LIBRARIES})
#set(LIBRARIES ${LIBRARIES} ${ROOFIT_LIBRARIES})

# BOOST, provides free peer-reviewed portable C++ source libraies
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.54.0 COMPONENTS system filesystem regex REQUIRED) 

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS}) 
endif()

# mvIMPACT Acquire SDK
find_package( mvIMPACT REQUIRED )
INCLUDE_DIRECTORIES( ${mvIMPACT_INCLUDE_DIRS} )
set( LIBRARIES ${LIBRARIES} ${mvIMPACT_LIBRARIES} )

# Find and include OpenCV directories
find_package(OpenCV REQUIRED)
if(${OpenCV_FOUND})
  MESSAGE(STATUS "OpenCV version: " ${OpenCV_VERSION})
  MESSAGE(STATUS "OpenCV libs: " ${OpenCV_LIBS})
endif(${OpenCV_FOUND})
include_directories(${OpenCV_INCLUDE_DIRS})

if(${APPLE})

  MESSAGE( STATUS "Building for Mac OS X, switching on C++11 flags for Mac OS X/clang" )

  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pedantic -Wall -Wextra")

endif(${APPLE})

IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

  MESSAGE( STATUS "Building for Linux, switching on C++11 flags for Linux/gcc" )

  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -pedantic -Wall -Wextra -O0 -ggdb")

ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

add_executable(WindingControl WindingControl.cpp WindingControlMain.cpp)
target_link_libraries(WindingControl ${mvIMPACT_LIBRARIES} ${OpenCV_LIBS})
