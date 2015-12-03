IF (ENABLE_QT5)
    QT5_WRAP_UI(UI_HEADERS gazergui.ui)
ENDIF()

SET(GAZETOOL_SRC
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
    SET(GAZETOOL_SRC ${GAZETOOL_SRC} yarpsupport.cpp)
ENDIF()


ADD_LIBRARY(gaze SHARED ${GAZETOOL_SRC})
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


IF (ENABLE_QT5)
    SET(GAZETOOL_BIN
        main.cpp
        gazergui.cpp
        workerthread.cpp
        glimageview.cpp
        ${UI_HEADERS})

    ADD_EXECUTABLE(gazetool ${GAZETOOL_BIN})
    ADD_DEPENDENCIES(gazetool ${UI_HEADERS})
    TARGET_LINK_LIBRARIES(gazetool gaze GL)
    qt5_use_modules(gazetool Core Widgets Gui OpenGL)

    INSTALL(TARGETS gazetool
      RUNTIME DESTINATION bin
    )
ENDIF()
