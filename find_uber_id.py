
from shapely.geometry import Point, Polygon
import random
import geojson

gj_file = '../pittsburgh_censustracts.json'

with open(gj_file) as f:
    gj = geojson.load(f)

polys = [(ind + 1, Polygon(gj[ind]['geometry']['coordinates'][0][0])) for ind in range(5)]

print(polys)

# Internal point from gridwise dataset
point = Point([-83.0072144, 39.9788975])

from operator import itemgetter

min_dist, min_id, min_poly = min(((poly.distance(point), ind, poly) for ind, poly in polys), key=itemgetter(0))
check = [(poly.distance(point), ind, poly) for ind, poly in polys]
print(check)

print(min_dist, min_id, min_poly)

