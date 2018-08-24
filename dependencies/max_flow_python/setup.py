from distutils.core import setup
import numpy as np
from distutils.extension import Extension

package_name = 'crf_max_flow'
module_name1 = 'max_flow'
module_name2 = 'max_flow3d'
module1 = Extension(module_name1,
                    include_dirs = [np.get_include()],
                    sources = ['max_flow.cpp', 'maxflow-v3.0/graph.cpp', 'maxflow-v3.0/maxflow.cpp'])
module2 = Extension(module_name2,
                    include_dirs = [np.get_include()],
                    sources = ['max_flow3d.cpp', 'maxflow-v3.0/graph.cpp', 'maxflow-v3.0/maxflow.cpp'])
setup(name=package_name,
      ext_modules = [module1, module2])
