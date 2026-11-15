cmake_minimum_required(VERSION 3.8)
project(igvc_localization)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# igvc_localization/ python paketini kur (dugumler birbirinden import edebilsin diye)
ament_python_install_package(${PROJECT_NAME})

# Calistirilabilir dugumler: "<executable>:<modul>" -> ros2 run igvc_localization <executable>
# Modul igvc_localization/<modul>.py, icinde main() olmali. Yeni dugum icin listeye bir satir ekle.
# lib/ altina modulu import eden kucuk bir baslatici kurulur; boylece
# --symlink-install ile .py degisiklikleri yeniden build gerektirmez.
set(PYTHON_NODES
  ekf_node:ekf_node                 # EKF (mode: local -> odom->base_footprint, global -> map->odom)
  gps_odometry:gps_odometry         # NavSatFix -> map (ENU) anten konumu, datum
  localization_eval:localization_eval  # simulasyon: gercek konuma karsi hata
)
foreach(entry ${PYTHON_NODES})
  string(REPLACE ":" ";" parts ${entry})
  list(GET parts 0 exe)
  list(GET parts 1 module)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/nodes_tmp/${exe}
    "#!/usr/bin/env python3\n"
    "import sys\n"
    "from ${PROJECT_NAME}.${module} import main\n"
    "sys.exit(main())\n")
  file(COPY ${CMAKE_CURRENT_BINARY_DIR}/nodes_tmp/${exe}
       DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/nodes
       FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/nodes/${exe} DESTINATION lib/${PROJECT_NAME})
endforeach()

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  # pip'le kurulan 'anyio' pytest eklentisi Ubuntu 22.04'un pytest 6.2.5'ini cokertir; kapat.
  set(PYTEST_ADDOPTS_FIX "-p no:anyio")
  set(ENV{PYTEST_ADDOPTS} "${PYTEST_ADDOPTS_FIX}")
  find_package(ament_cmake_pytest REQUIRED)
  ament_has_pytest(HAS_PYTEST QUIET)
  if(HAS_PYTEST)
    foreach(t test_geodesy test_ekf_core)
      ament_add_pytest_test(${t} test/${t}.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        ENV "PYTEST_ADDOPTS=${PYTEST_ADDOPTS_FIX}")
    endforeach()
  else()
    message(STATUS "pytest calismiyor, igvc_localization testleri atlaniyor (sudo apt install python3-pytest)")
  endif()
endif()

ament_package()
