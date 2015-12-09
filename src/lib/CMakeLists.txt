cmake_minimum_required(VERSION 2.8)

PROJECT(gaze_lib)

set(GAZE_LIB_MAJOR_VERSION 0)
set(GAZE_LIB_MINOR_VERSION 1)
set(GAZE_LIB_PATCH_VERSION 0)
set(GAZE_LIB_VERSION
  ${GAZE_LIB_MAJOR_VERSION}.${GAZE_LIB_MINOR_VERSION}.${GAZE_LIB_PATCH_VERSION})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

OPTION(ENABLE_YARP_SUPPORT "Enable yarp support" OFF)
OPTION(ENABLE_QT5 "Build demo (QT5 is required)" ON)

set(CMAKE_AUTOMOC TRUE)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

FIND_PACKAGE(dlib REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Boost COMPONENTS system chrono program_options REQUIRED)

IF (ENABLE_QT5)
    FIND_PACKAGE(Qt5Core REQUIRED)
    FIND_PACKAGE(Qt5Widgets REQUIRED)
    FIND_PACKAGE(Qt5OpenGL REQUIRED)
    FIND_PACKAGE(Qt5Gui REQUIRED)
    ADD_DEFINITIONS(-DENABLE_QT5)
ENDIF()

IF(ENABLE_YARP_SUPPORT)
    FIND_PACKAGE(YARP REQUIRED)
ENDIF()

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} ${YARP_INCLUDE_DIRS} ${dlib_INCLUDE_DIRS})

SET(GAZE_LIB_SRC
    imageprovider.cpp
    faceparts.cpp
    pupilfinder.cpp
    eyelidlearner.cpp
    mutualgazelearner.cpp
    relativeeyelidlearner.cpp
    relativegazelearner.cpp
    verticalgazelearner.cpp
    facedetectionworker.cpp
    shapedetectionworker.cpp
    gazehyps.cpp
    regressionworker.cpp
    eyepatcher.cpp
    featureextractor.cpp
    abstractlearner.cpp
    rlssmoother.cpp
    blockingqueue.h
)

IF(ENABLE_YARP_SUPPORT)
    ADD_DEFINITIONS(-DENABLE_YARP_SUPPORT)
    SET(GAZE_LIB_SRC ${GAZE_LIB_SRC} yarpsupport.cpp)
ENDIF()


ADD_LIBRARY(gaze SHARED ${GAZE_LIB_SRC})
target_link_libraries(gaze ${OpenCV_LIBS} ${dlib_LIBRARIES} ${Boost_LIBRARIES} ${YARP_LIBRARIES})

IF (ENABLE_QT5)
    qt5_use_modules(gaze Core)
ENDIF()


export(TARGETS gaze
  FILE "${PROJECT_BINARY_DIR}/gazeTargets.cmake")
export(PACKAGE gaze)


set(INSTALL_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include/gaze/")

configure_file("config/gazeConfig.cmake.in"
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/gazeConfig.cmake" @ONLY)
configure_file("config/gazeConfigVersion.cmake.in"
  "${PROJECT_BINARY_DIR}/gazeConfigVersion.cmake" @ONLY)

export(TARGETS gaze
  FILE "${PROJECT_BINARY_DIR}/gazeTargets.cmake")

INSTALL(TARGETS gaze
     LIBRARY DESTINATION lib
)

INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION "include/gaze"
    FILES_MATCHING
    PATTERN "*.h")

install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/gazeConfig.cmake"
  "${PROJECT_BINARY_DIR}/gazeConfigVersion.cmake"
  "${PROJECT_BINARY_DIR}/gazeTargets.cmake"
  DESTINATION "lib/cmake/gaze" COMPONENT dev)
