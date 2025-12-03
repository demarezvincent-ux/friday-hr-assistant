{ pkgs }: {
  deps = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.flask
    pkgs.run
  ];
  env = {
    PYTHON_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      # Needed for many Python native libraries
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
    ];
  };
}