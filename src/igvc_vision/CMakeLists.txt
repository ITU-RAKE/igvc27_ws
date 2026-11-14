cmake_minimum_required(VERSION 3.8)
project(igvc_vision)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# igvc_vision/ python paketini kur (dugumler birbirinden import edebilsin diye)
ament_python_install_package(${PROJECT_NAME})

# Calistirilabilir dugumler: "<executable>:<modul>" -> ros2 run igvc_vision <executable>
# Modul igvc_vision/<modul>.py, icinde main() olmali. Yeni dugum icin listeye bir satir ekle.
# lib/ altina modulu import eden kucuk bir baslatici kurulur; boylece
# --symlink-install ile .py degisiklikleri yeniden build gerektirmez.
set(PYTHON_NODES
  lidar_map:lidar_map               # anlik lidar yerel haritasi
  surround_view:surround_view_node  # 4 kamera 360 deg kus bakisi (geometriden homografi)
  four_cam_test:four_cam_test       # kus bakisi + YOLO segmentasyon (TensorRT)
  camera_map:camera_map             # kus bakisinda kirmizi -> engel occupancy grid
  fused_map:fused_map               # lidar + kamera haritalarini birlestir
)
foreach(entry ${PYTHON_NODES})
  string(REPLACE ":" ";" parts ${entry})
  list(GET parts 0 exe)
  list(GET parts 1 module)
  # Once gecici dosyaya yaz, sonra calistirma izniyle kopyala (symlink-install izni korur)
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
  # pip ile kurulan 'anyio' (httpx/openai bagimliligi) pytest'e otomatik yuklenen bir eklenti
  # kaydeder; bu eklenti pytest >= 7 ister ve Ubuntu 22.04'un pytest 6.2.5'ini acilista
  # cokertir (ament bunu "pytest bulunamadi" diye raporlar). Eklentiyi devre disi birak:
  # hem ament'in configure sirasindaki pytest kontrolu hem de colcon test icin.
  # anyio eklentisi olmayan makinelerde etkisizdir.
  set(PYTEST_ADDOPTS_FIX "-p no:anyio")
  set(ENV{PYTEST_ADDOPTS} "${PYTEST_ADDOPTS_FIX}")
  find_package(ament_cmake_pytest REQUIRED)
  ament_has_pytest(HAS_PYTEST QUIET)
  if(HAS_PYTEST)
    ament_add_pytest_test(test_surround_geometry test/test_surround_geometry.py
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      ENV "PYTEST_ADDOPTS=${PYTEST_ADDOPTS_FIX}")
    ament_add_pytest_test(test_costmap test/test_costmap.py
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      ENV "PYTEST_ADDOPTS=${PYTEST_ADDOPTS_FIX}")
  else()
    message(STATUS "pytest calismiyor, igvc_vision testleri atlaniyor (sudo apt install python3-pytest)")
  endif()
endif()

ament_package()
