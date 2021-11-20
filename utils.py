import time
import csv


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
    

if __name__ == '__main__':
    read_input('download/Gridwise/gridwise_trips_sample_pit.csv')

