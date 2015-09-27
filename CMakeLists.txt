cmake_minimum_required(VERSION 2.8)
#cmake_policy(SET CMP0043 OLD)
#cmake_policy(SET CMP0046 OLD)

PROJECT(dlibgazer)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

SET(DLIB_DIR "dlib" CACHE PATH "dlib directory")
INCLUDE(${DLIB_DIR}/dlib/cmake)

OPTION(ENABLE_YARP_SUPPORT "Enable yarp support" OFF)
set(CMAKE_AUTOMOC TRUE)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Boost COMPONENTS system chrono program_options REQUIRED)
FIND_PACKAGE(Qt5Core REQUIRED)
FIND_PACKAGE(Qt5Widgets REQUIRED)
FIND_PACKAGE(Qt5OpenGL REQUIRED)
FIND_PACKAGE(Qt5Gui REQUIRED)

IF(ENABLE_YARP_SUPPORT)
    FIND_PACKAGE(YARP REQUIRED)
ENDIF()

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIR} ${Boost_INCLUDE_DIR} ${YARP_INCLUDE_DIRS})

ADD_SUBDIRECTORY(src)
#ADD_SUBDIRECTORY(inpainttool)

CONFIGURE_FILE(dlibgazer.sh.in dlibgazer.sh @ONLY)

INSTALL(FILES
    data/shape_predictor_68_face_landmarks.dat
    data/gaze_est_deg.dat
    data/lid_est.dat
    data/vertgaze_est_deg.dat
    DESTINATION share/dlibgazer)

INSTALL(PROGRAMS ${CMAKE_BINARY_DIR}/dlibgazer.sh DESTINATION bin)
