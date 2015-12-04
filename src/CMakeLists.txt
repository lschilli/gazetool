QT5_WRAP_UI(UI_HEADERS gazergui.ui)

INCLUDE_DIRECTORIES(lib)

IF(ENABLE_YARP_SUPPORT)
    ADD_DEFINITIONS(-DENABLE_YARP_SUPPORT)
    SET(GAZETOOL_SRC ${GAZETOOL_SRC} yarpsupport.cpp)
ENDIF()

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


CONFIGURE_FILE("${GAZE_CONFIG_DIR}/gazetool.sh.in" gazetool.sh @ONLY)
MESSAGE( STATUS "${GAZE_CONFIG_DIR}")

INSTALL(TARGETS gazetool
  RUNTIME DESTINATION bin
)

INSTALL(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/gazetool.sh DESTINATION bin)
