cmake_minimum_required(VERSION 3.5)

project(convolution LANGUAGES C)

add_executable(convolution main.c
    stb_image.h
    stb_image_write.h)

include(GNUInstallDirs)
install(TARGETS convolution
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
