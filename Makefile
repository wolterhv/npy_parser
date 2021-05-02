test:
	g++ -g `pkgconf --cflags --libs eigen3` -o npy_parser_test npy_parser.cpp test.cpp
