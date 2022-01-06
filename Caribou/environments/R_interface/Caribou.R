# /usr/bin/R
library(optparse)
library(reticulate)

# CLI options
option_list = list(
  make_option(c("-e","--env"), type = "character", default = NULL, help = "Python virtual environment folder location on disk"),
  make_option(c("-c","--config"), type = "character", default = NULL, help = "Config file location on disk"))
opt_parser = OptionParser(option_list=option_list)
argv = parse_args(opt_parser)

# Define system's python engine
if (virtualenv_exists(argv$env)){
  # If using virtual environment
  use_virtualenv(argv$env, required = T)
  if (! py_module_available("Caribou")){
    py_install("../../../../Caribou/", argv$env, method = "virtualenv", pip = T)
  }
  print("Using virtual environment for python binaries")
} else {
  # If no virtual environment is given
  python = Sys.which("python3")
  use_python(python)
  if (! py_module_available("Caribou")){
    py_install("../../../../Caribou/", pip = T)
  }
  print("Using system default python binaries")
}

caribou = import("Caribou")
caribou_args = r_to_py(c("Caribou.py",argv$config), convert = T)
caribou$Caribou$caribou(caribou_args)
