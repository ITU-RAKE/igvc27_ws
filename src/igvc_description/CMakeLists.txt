cmake_minimum_required(VERSION 3.8)
project(igvc_description)

find_package(ament_cmake REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Govde mesh'i igvc_macros.xacro'daki olculerden uretilir (kaynakta STL tutulmaz).
# Macros degisince bir sonraki build'de otomatik yeniden uretilir.
set(CHASSIS_STL ${CMAKE_CURRENT_BINARY_DIR}/meshes/chassis.stl)
add_custom_command(
  OUTPUT ${CHASSIS_STL}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_chassis_mesh.py
          --macros ${CMAKE_CURRENT_SOURCE_DIR}/urdf/igvc_macros.xacro ${CHASSIS_STL}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/urdf/igvc_macros.xacro
          ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_chassis_mesh.py
  COMMENT "chassis.stl igvc_macros.xacro'dan uretiliyor"
)
add_custom_target(chassis_mesh ALL DEPENDS ${CHASSIS_STL})

install(
  DIRECTORY urdf launch rviz worlds config models meshes_cad
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES ${CHASSIS_STL}
  DESTINATION share/${PROJECT_NAME}/meshes
)

install(
  PROGRAMS scripts/generate_chassis_mesh.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
