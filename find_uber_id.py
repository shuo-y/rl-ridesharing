"""
Mapping the polygon id from Gridwise data
to the polygon id from Uber data
"""

import shapely
from shapely.geometry import Point, Polygon
import random
import geojson
import csv

gj_file = 'download/pittsburgh_censustracts.json'

with open(gj_file) as f:
    gj = geojson.load(f)



polys = [(ind + 1, Polygon(gj[ind]['geometry']['coordinates'][0][0])) for ind in range(608)]


# Internal point from gridwise dataset
point = Point([-83.0072144, 39.9788975])

from operator import itemgetter

min_dist, min_id, min_poly = min(((poly.distance(point), ind, poly) for ind, poly in polys), key=itemgetter(0))
#check = [(poly.distance(point), ind, poly) for ind, poly in polys]
#print(check)

#print(min_dist, min_id, min_poly)

import tempfile

def write_geojson(name, features):
   # feature is a shapely geometry type
   geom_in_geojson = geojson.Feature(geometry=features, properties={})
   
   # tmp_file = tempfile.mkstemp(suffix='.geojson')
   with open(name + '.geojson', 'w') as outfile:
      geojson.dump(geom_in_geojson, outfile)

import json
def write_json(name, data):
    with open(name, 'w') as jsonf:
        json.dump(data, jsonf)

import shapely.wkt
grid_csv_file = 'download/Gridwise/gridwise_trips_sample_block_groups.csv'
print('GridID,UberID,dist')
with open(grid_csv_file) as f:
    reader = csv.DictReader(f, delimiter=',')
    for row in reader:
        #print(row)
        #print(row['GEOID'])
        #print(type(row['geom_internal_point']))
        point = shapely.wkt.loads(row['geom_internal_point'])
        min_dist, uber_min_id, min_poly = min(((poly.distance(point), ind, poly) for ind, poly in polys), key=itemgetter(0))
        #print(min_poly)
        record = '%s,%s,%s' % (row['GEOID'], uber_min_id, min_dist)
        grid_poly = shapely.wkt.loads(row['geom'])
        write_geojson('%s-gridwise_gid-%s' % (uber_min_id, row['GEOID']), grid_poly)
        write_geojson('%s-uber_gid' % uber_min_id, min_poly)
        print(record)
        

