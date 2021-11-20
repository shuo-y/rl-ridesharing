import time
import csv
from graph_tool import Graph
import shapely
import geojson
from shapely.geometry import Point, Polygon
import numpy as np

def parse_time(strtime: str):
    timeformat = '%Y-%m-%d %H:%M:%S UTC'
    try:
        request_time = time.strptime(strtime, timeformat)
        return request_time
    except ValueError:
        return ''
        
def read_for_map(reader: csv.DictReader, grid2uber: dict):
    timemap = {}
    for row in reader:
        request_time = parse_time(row['request_time'])
        start_time = parse_time(row['start_time'])
        end_time = parse_time(row['end_time'])
        #print(row['duration'])
        if (start_time != '' and  end_time != '' 
            and row['start_block_group'] != '' and row ['end_block_group'] != ''):
            dur = (time.mktime(end_time) - time.mktime(start_time))
            timestep = start_time.tm_hour * 3600 + start_time.tm_min * 60 + start_time.tm_sec
            if timestep not in timemap:
                timemap[timestep] = []
                
            src = int(grid2uber[row['start_block_group']])
            dst = int(grid2uber[row['end_block_group']])

            earnings = int(100 * float(row['driver_total_earnings'])) 
            trip = {'src': src, 'dst': dst, 'dur': int(dur), 'earnings': earnings}
            timemap[timestep].append(trip)

    return timemap



def read_input(filename):
    """
    Read input files from Gridwise csv
    """
    timemap = {}

    grid2uber = {}
    uber2grid = {}
    with open('mapping.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
           grid2uber[row['GridID']] = row['UberID']
           uber2grid[row['UberID']] = row['GridID']


    with open(filename, 'r') as f:
        total = f.readlines()
        print('Total %s records' % (len(total) - 1))
        train_set = total[:int(len(total) * 0.7)]
        print('Train %s records' % (len(train_set) - 1))
        reader = csv.DictReader(train_set)
        train_data = read_for_map(reader, grid2uber)
        dev_set = total[0:1] + total[int(len(total) * 0.7):]
        print('Dev %s records' % (len(dev_set) - 1))
        reader = csv.DictReader(dev_set)     
        dev_data = read_for_map(reader, grid2uber)
        return train_data, dev_data


def construct_graph(filename):
    """
    Construct graph for the city
    """
    alldaygraphs = []
    nodes = 608
    weights = []

    for i in range(24):
        g = Graph()
        g.add_vertex(nodes + 1)
        eweight = g.new_ep('double')
        alldaygraphs.append(g)
        weights.append(eweight)

    """
    Use graph_tool 
    See https://graph-tool.skewed.de/static/doc/index.html
    """
   
    hours = [[] for _ in range(24)]
    minno = 1000
    maxno = 0
    minh = 1000
    maxh = -1
    with open(filename, 'r') as f:
        total = f.readlines()
        reader = csv.DictReader(total)
        for row in reader:
            src = int(row['sourceid'])
            dst = int(row['dstid'])
            minno = min(minno, src)
            minno = min(minno, dst)
            maxno = max(maxno, src)
            maxno = max(maxno, dst)
            hod = int(row['hod'])
            maxh = max(maxh, hod) 
            minh = min(minh, hod)
            duration = float(row['mean_travel_time'])
            g = alldaygraphs[hod]
            ew = weights[hod]
            # Make sure no duplicate edges
            assert g.edge(src, dst) == None 
            g.add_edge(src, dst)
            e = g.edge(src, dst)
            ew[e] = duration
    print('Min node %s Max node %s minh %s maxh %s' % (minno, maxno, minh, maxh))

    return alldaygraphs, weights
    

def get_time_hod(src, dst, hod, alldaygraphs, weights):
    e = alldaygraphs[hod].edge(src, dst)
    durs = [weights[hours][alldaygraphs[hours].edge(src, dst)] if alldaygraphs[hours].edge(src, dst) != None else None for hours in range(24)]
    durs = [dur for dur in durs if dur]
    if e != None:
        dur = weights[hod][e]
        pass
    elif len(durs) > 0:
       dur = np.mean(durs)
    else:
        print("%s ->  %s hod %s not found" % (src, dst, hod)) 


def check_time(src, dst, alldaygraphs, weights):
    """
    check that the duration data between src and dst exists
    """
    for hod, graph in enumerate(alldaygraphs):
        get_time_hod(src, dst, hod, alldaygraphs, weights)

def city_graph(gj_file, alldaygraphs, weights):

    with open(gj_file, 'r') as f:
        gj = geojson.load(f)
    polys = [(ind + 1, Polygon(gj[ind]['geometry']['coordinates'][0][0])) for ind in range(608)]
    print(gj[0]['geometry']['coordinates'][0][0])
    print(list(polys[0][1].exterior.coords))
    print(polys[0][1].area)

    neighbors = {}
    for cur_poly in polys:
        neighbors[cur_poly[0]] = []
        for cnt, ply in enumerate(polys):
            if not cur_poly[1].disjoint(ply[1]) and cur_poly[0] != ply[0]:
                neighbors[cur_poly[0]].append(ply[0])
                check_time(cur_poly[0], ply[0], alldaygraphs, weights)

    return neighbors


if __name__ == '__main__':
    #read_input('download/Gridwise/gridwise_trips_sample_pit.csv')
    graphs, weights = construct_graph('download/pittsburgh-censustracts-2020-1-All-HourlyAggregate.csv')
    city_graph('download/pittsburgh_censustracts.json', graphs, weights)
