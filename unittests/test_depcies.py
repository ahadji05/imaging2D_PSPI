# All modules (that do not belong to Standard Python Library) should
# be installed by the user before running this code!
# Currently these are: numpy, scipy and pandas.
#-------------------------------------------------------------------

def print_err( mod_name):
    print( "Module "+str(mod_name)+" not found, please install it before running the tests!")


try:
    import numpy
except ModuleNotFoundError:
    print_err( "numpy" )
    exit()

try:
    import scipy
except ModuleNotFoundError:
    print_err( "scipy" )
    exit()

try:
    import pandas
except ModuleNotFoundError:
    print_err( "pandas" )
    exit()


print("All necessary python modules found!")
