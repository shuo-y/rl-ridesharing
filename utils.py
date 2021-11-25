import time
import csv
import numpy as np

def parse_time(strtime: str):
    timeformat = '%Y-%m-%d %H:%M:%S UTC'
    try:
        request_time = time.strptime(strtime, timeformat)
        return request_time
    except ValueError:
        return ''
        
def read_for_map(reader: csv.DictReader, grid2uber: dict, interval=300):
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
               
            src = int(grid2uber[row['start_block_group']])
            dst = int(grid2uber[row['end_block_group']])
            
            assert src != 0
            assert dst != 0

            earnings = int(100 * float(row['driver_total_earnings'])) 
            trip = {'src': src, 'dst': dst, 'dur': int(dur), 'earnings': earnings}
            # Orders inside the same interval are put into the same map
            timestep = timestep//interval
            if timestep not in timemap:
                timemap[timestep] = []
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
    Construct graph for travel times around the city
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
    

def get_time(src, dst, graph, weight):
    e = graph.edge(src, dst)
    direct = 1
    if e != None:
        dur = weight[e]
    else:
        vlist, elist  = topology.shortest_path(graph, src, dst, weight)
        direct = 0
        if len(elist) == 0:
            direct = -1
            dur = -1
        else:
            dur = np.sum([weight[e] for e in elist])
    return dur, direct

def check_time(alldaygraphs, weights):
    """
    check that the duration data between src and dst exists
    """
    direct_cnt = 0
    not_reach = 0
    print('hod,src,dst,dur,type')
    for src in range(1, 609):
        for dst in range(1, 609):
            #print('\rcheck nodes src %s dst %s...' % (src, dst))
            if src == dst:
                continue
            for hod in range(0, 24):
                dur, cnt = get_time(src, dst, alldaygraphs[hod], weights[hod])
                if cnt == 1:
                    data_type = "direct"
                    direct_cnt += 1
                elif cnt == -1:
                    data_type = "not_reach"
                    not_reach += 1
                else:
                    data_type = "path"
                print('%d,%d,%d,%f,%s' % (hod, src, dst, dur, data_type))
    print('direct %s not reach %s total %s' % (direct_cnt, not_reach,  24 * 609 * 609))

def process_hourly_data(filename):
    """
    no need to build graph
    simply check if src->dst at a time exists
    """
    hourly_map = [{} for hod in range(24)]
    with open(filename, 'r') as f:
        reader = csv.DictReader(f.readlines())
        for row in reader:
            src = int(row['sourceid'])
            dst = int(row['dstid'])
            hod = int(row['hod'])
            if src not in hourly_map[hod]:
                hourly_map[hod][src] = {}
            if dst not in hourly_map[hod][src]:
                hourly_map[hod][src][dst] = float(row['mean_travel_time'])
            else:
                hourly_map[hod][src][dst] = min(hourly_map[hod][src][dst], float(row['mean_travel_time']))
    return hourly_map


def process_weekly_data(filename):
    """
    Build src dst map and build graph
    if duplicate use the minimum of time
    """
    allday_map = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f.readlines())
        for row in reader:
            src = int(row['sourceid'])
            dst = int(row['dstid'])
            if src not in allday_map:
                allday_map[src] = {}
            if dst not in allday_map[src]:
                allday_map[src][dst] = float(row['mean_travel_time'])
            else:
                allday_map[src][dst] = min(allday_map[src][dst], float(row['mean_travel_time']))
    # Build ud graph for time
    allday_ug = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f.readlines())
        for row in reader:
            src = int(row['sourceid'])
            dst = int(row['dstid'])
            if src > dst:
                src, dst = dst, src
          
            if src not in allday_ug:
                allday_ug[src] = {}
            if dst not in allday_ug[src]:
                allday_ug[src][dst] = float(row['mean_travel_time'])
            else:
                allday_ug[src][dst] = min(allday_ug[src][dst], float(row['mean_travel_time']))
    
    g = Graph(directed=False)
    all_edges = [(src, dst, allday_ug[src][dst]) for src in allday_ug for dst in allday_ug[src]]
    g.add_vertex(609) # 608 nodes + 1 to use index directly
    eweight = g.new_ep("double")
    g.add_edge_list(all_edges, eprops=[eweight])

    from graph_tool.topology import label_components
    comp, hist = label_components(g)
    print("COMPONENTS%s" % len(hist))
    return allday_map, g, eweight

def find_closest_path(src, dst, allday_time, polys):
    """
    polys is list reading from gj_file
    """
    
    dst_cand = [(polys[dst - 1].distance(polys[i]), i + 1) for i in range(len(polys))]
    src_cand = [(polys[src - 1].distance(polys[i]), i + 1) for i in range(len(polys))]
    dst_cand.sort(key=lambda x : x[0])
    src_cand.sort(key=lambda x : x[0])

    mindur = 3601 * 24 # a long time
    for hops in range(1, len(dst_cand) + len(src_cand)):
        mindur = 3601 * 24
        for src_ind in range(hops):
            dst_ind = hops - src_ind
            nsrc = src_cand[src_ind][1]
            ndst = dst_cand[dst_ind][1]
            if allday_time[nsrc][ndst] > 0:
                dur = allday_time[nsrc][ndst]
                mindur = min(mindur, dur)
            if allday_time[ndst][nsrc] > 0:
                dur = allday_time[ndst][nsrc]
                mindur = min(mindur, dur)
        # keep hops as small as possible
        if mindur != 3601 * 24:
            return mindur, hops 
     # should find a neighbor of the dst
    # and there is a path from src to the neighbor

    if mindur == 3601 * 24:
        print('not time from %s to %s ' % (src, dst))
        assert False
    return mindur

def save_travel_time(allday_map, allday_graph, eweight, polys):
    """ 
    if no direct path, find if there is a path
    if no path, find the closest polygon of the target block
    """
    from graph_tool.draw import graph_draw
    from graph_tool import GraphView
    #u = GraphView(allday_graph, efilt=lambda e: eweight[e] > 2000 and eweight[e] < 2010)
    #u = Graph(u, prune=True)
    #graph_draw(u, output='allday_part.svg')
    #exit(0)
    allday_time = np.zeros((609, 609))
    for src in range(1, 609):
        for dst in range(1, 609):
            if src == dst:
                continue
            if src in allday_map and dst in allday_map[src]:
                allday_time[src][dst] = allday_map[src][dst]
                print('%d,%d,%f,%s' % (src, dst, allday_time[src][dst], 'direct'))
            elif dst in allday_map and src in allday_map[dst]:
                # use the reverse direction
                allday_time[src][dst] = allday_map[dst][src]
                print('%d,%d,%f,%s' % (src, dst, allday_time[src][dst], 'rev'))
            else:
                vlist, elist = topology.shortest_path(allday_graph, src, dst, eweight)
                if len(elist) > 0:
                    allday_time[src][dst] = np.sum([eweight[e] for e in elist])
                    print('%d,%d,%f,%s' % (src, dst, allday_time[src][dst], 'directpath'))
                    continue
                
                vlist, elist = topology.shortest_path(allday_graph, dst, src, eweight)
                if len(elist) > 0:
                    allday_time[src][dst] = np.sum([eweight[e] for e in elist])
                    print('%d,%d,%f,%s' % (src, dst, allday_time[src][dst], 'revpath'))
                    continue


    for src in range(1, 609):
        for dst in range(1, 609):
            if src == dst:
                continue
            if allday_time[src][dst] > 0 :
                continue
            dur, hops = find_closest_path(src, dst, allday_time, polys)
            allday_time[src][dst] = dur
            print('%d,%d,%f,%s,%d' % (src, dst, allday_time[src][dst], 'estpath', hops))
    return allday_time

def city_graph(gj_file):

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

    return neighbors

import pickle
if __name__ == '__main__':
    import shapely
    import geojson
    from graph_tool import Graph, topology
    from shapely.geometry import Point, Polygon
    #hourly_map = process_hourly_data('download/pitt2019-2020q1Allhour.csv')
    #pickle.dump(hourly_map, open('hourly_map', 'wb'))
    maps, graphs, weights = process_weekly_data('download/pitt2019-2020q1weekly.csv')
    gj = geojson.load(open('download/pittsburgh_censustracts.json'))
    polys = [Polygon(gj[ind]['geometry']['coordinates'][0][0]) for ind in range(608)]
    allday_time = save_travel_time(maps, graphs, weights, polys)
    pickle.dump(allday_time, open('allday_time', 'wb'))
    #read_input('download/Gridwise/gridwise_trips_sample_pit.csv')
    #city_graph('download/pittsburgh_censustracts.json', graphs, weights)
