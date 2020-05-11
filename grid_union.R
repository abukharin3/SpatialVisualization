library('geojsonio')
library("RcppCNPy")

#Geojson with numbers
path = "data/grids.json"

# Zone configuration
fmat <- npyLoad("data/grid-Jan-APR-2019-PD.npy")
zone <- fmat[,2]

grid <- geojsonio::geojson_read(path, what = "sp")
grid$zone <- zone

# Merge polygons by ID
grid.union <- unionSpatialPolygons(grid, grid$zone)
gridjson <- geojson_json(grid.union)

# For some reason there is an error and it may save as myfile.geojson
geojson_write(grid.union, "data/merged_grids_beats=15.json")
