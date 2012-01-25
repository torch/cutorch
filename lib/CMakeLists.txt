FILE(GLOB _all_files *)

FOREACH(_file ${_all_files})
  IF(EXISTS "${_file}/CMakeLists.txt")
    ADD_SUBDIRECTORY(${_file})
  ENDIF(EXISTS "${_file}/CMakeLists.txt")
ENDFOREACH(_file ${_all_files})
