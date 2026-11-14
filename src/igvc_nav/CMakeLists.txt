cmake_minimum_required(VERSION 3.8)
project(igvc_nav)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# igvc_nav/ python paketini kur (dugumler birbirinden import edebilsin diye)
ament_python_install_package(${PROJECT_NAME})

# Calistirilabilir dugumler: "<executable>:<modul>" -> ros2 run igvc_nav <executable>
# Yeni dugum eklerken bu listeye bir satir ekle.
# lib/ altina modulu import eden kucuk bir baslatici kurulur; boylece
# --symlink-install ile .py degisiklikleri yeniden build gerektirmez.
set(PYTHON_NODES
  path_planner:path_planner         # fused_map uzerinde hedef secimi + A* yerel yol
  path_tracker:path_tracker         # planned_path'i pure pursuit ile takip -> /cmd_vel
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

ament_package()
