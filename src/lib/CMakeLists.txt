SET(GAZETOOL_LIB_SRC
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
    SET(GAZETOOL_LIB_SRC ${GAZETOOL_LIB_SRC} yarpsupport.cpp)
ENDIF()


ADD_LIBRARY(gaze SHARED ${GAZETOOL_LIB_SRC})
target_link_libraries(gaze ${OpenCV_LIBS} ${dlib_LIBRARIES} ${Boost_LIBRARIES} ${YARP_LIBRARIES})

IF (ENABLE_QT5)
    qt5_use_modules(gaze Core)
ENDIF()

INSTALL(TARGETS gaze
     LIBRARY DESTINATION lib
)

INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION "include/gaze"
    FILES_MATCHING
    PATTERN "*.h")
