# @author: Mohammed Mehboob

from heapq import heappush, heappop
from PIL import Image, ImageDraw
from math import sqrt, exp
import sys

# ## Setting up Object Oritented Design ============================================

class Pixel:
    
    def __init__(self, x,y,elevation,terrain_cost):
        self.x = x
        self.y = y
        self.elevation = elevation
        self.terrain_cost = terrain_cost
    
    def get_successors(self):
        global data
        
        successors = dict()
        
        top_row = self.y == 1 - 1
        rightmost_column = self.x == 395 - 1
        leftmost_column = self.x == 1 - 1
        bottom_row = self.y == 500 - 1
        
        if not top_row:
            successors['N'] = data[self.y-1][self.x]
            
        if not rightmost_column:
            successors['E'] = data[self.y][self.x+1]        
        
        if not leftmost_column:
            successors['W'] = data[self.y][self.x-1]
        
        if not bottom_row:
            successors['S'] = data[self.y+1][self.x]
        
        if not top_row and not leftmost_column:
            successors['NW'] = data[self.y-1][self.x-1]
            
        if not top_row and not rightmost_column:
            successors['NE'] = data[self.y-1][self.x+1]
            
        if not bottom_row and not leftmost_column:
            successors['SW'] = data[self.y+1][self.x-1]
        
        if not bottom_row and not rightmost_column:
            successors['SE'] = data[self.y+1][self.x+1]
                       
        return [x[1] for x in successors.items()]
    
    def movement_cost(self, successor):
        
        dx = ( successor.x - self.x ) * X_METERS
        dy = ( successor.y - self.y ) * Y_METERS
        dz = successor.elevation - self.elevation
        
        dx2 = pow(dx,2)
        dy2 = pow(dy,2)
        
        d = sqrt( dx2 + dy2 )
        m = dz/d
        
        tobler = 6 * exp(0.05-m) if abs(m)<10000 else 1
        euclidean_distance = sqrt( dx2 + dy2 + pow(dz,2) )
        return euclidean_distance * tobler * successor.terrain_cost 
    
    def distance(self, node):
        
        dx = ( node.x - self.x ) * X_METERS
        dy = ( node.y - self.y ) * Y_METERS
        dz = node.elevation - self.elevation
        
        # euclidean distance
        return sqrt( pow(dx,2) + pow(dy,2) + pow(dz,2) )

    def __lt__(self, other):
        return cost[self]<cost[other]
    
    def __repr__(self):
        return f'Pixel( x:{self.x}, y:{self.y}, e:{self.elevation})'


# ## Searching: ===============================================================================


def heuristic(goal,node):
    #corrected manhattan distance
    return ( abs( node.x - goal.x )*X_METERS + abs( node.y - goal.y )*Y_METERS + abs( node.elevation - goal.elevation ) )

def a_star(start, goal):
    global cost
    
    queue = [(0,start)]
    backtrace = { start:None }
    cost = { start:0 }
    
    while len(queue)>0:
        node = queue.pop(0)[1]
        
        if node == goal:
            backtrace_path = [node]
            came_before = backtrace[node]
            while came_before != None:
                backtrace_path.insert(0, came_before)
                came_before = backtrace[came_before]
            return backtrace_path
        
        for successor in node.get_successors():
            new_cost = cost[node] + node.movement_cost(successor)
            if successor not in cost or new_cost < cost[successor]:
                cost[successor] = new_cost
                priority = new_cost + heuristic(goal,successor)
                heappush(queue, (priority,successor))
                backtrace[successor]=node


# ## Graphics Stuff ===============================================================================

def draw_line(start,end):
    global image
    
    draw = ImageDraw.Draw(image)
#     px[start.x-1, start.y-1]=(255,0,0,255)
    
    p1 = ( start.x-1, start.y-1 )
    p2 = ( end.x-1, end.y-1 )
    draw.line( [p1,p2], fill = (128,0,0), width = 2 )
    draw.line( [p1,p2], fill = (255,0,0), width = 1 )

def highlight_checkpoints(trail):
    global image

    draw = ImageDraw.Draw(image)
    for checkpoint in trail:
        cx = checkpoint[0]
        cy = checkpoint[1]
        w = 2.5
        draw.rectangle([(cx-w, cy-w), (cx+w, cy+w)], fill=(0, 192, 192), outline=(60, 60, 60))    


# ## Solution ====================================================================================

def solution():    
    optimal_path = []
    
    for i in range(len(trail)-1):
    
        start_x = trail[i][0]
        start_y = trail[i][1]

        start = data[start_y][start_x]

        end_x = trail[i+1][0]
        end_y = trail[i+1][1]

        goal = data[end_y][end_x]
        
        optimal_subpath = a_star(start,goal)
        optimal_path = optimal_path + optimal_subpath
        
    path_length = 0
    #calculate path length
    #and draw lines
    for i in range(len(optimal_path)-1):
        begin = optimal_path[i]
        end = optimal_path[i+1]
        
        path_length = path_length + begin.distance(end)
        draw_line(begin,end)
    
    print(f"Total Path Length: {path_length}")

if __name__ == '__main__':

    # Terrain ===============================

    terrain_costs = dict()
    X_METERS = 10.29
    Y_METERS = 7.55

    OPEN_LAND = (248,148,18)
    ROUGH_MEADOW = (255,192,0)
    EASY_MOVEMENT_FOREST = (255,255,255)
    SLOW_RUN_FOREST = (2,208,60)
    WALK_FOREST = (2,136,40)
    IMPASSIBLE_VEG = (5,73,24)
    LAKE = (0,0,255)
    ROAD = (71,51,3)
    FOOTPATH = (0,0,0)
    OUT_OF_BOUNDS = (205,0,101)

    terrain_costs[FOOTPATH] = 1
    terrain_costs[ROAD] = 1.3
    terrain_costs[OPEN_LAND] = 1.5
    terrain_costs[EASY_MOVEMENT_FOREST] = 1.55
    terrain_costs[SLOW_RUN_FOREST] = 1.7
    terrain_costs[WALK_FOREST] = 2
    terrain_costs[ROUGH_MEADOW] = 2.5
    terrain_costs[LAKE] = 15
    terrain_costs[IMPASSIBLE_VEG] = 20
    terrain_costs[OUT_OF_BOUNDS] = 100

    #IO, and things ===============================

    terrain_image = sys.argv[1]
    elevation_file = sys.argv[2]
    trail_file = sys.argv[3]
    output = sys.argv[4]
    px = None
    
    trail = []
    with open(trail_file, 'r') as file:
        for line in file:
            trail.append(tuple(map(int,line.split() )))
    
    #Load in image =====================================

    LINES = 500
    VALUES = 395

    data = dict(dict())
    image = Image.open(terrain_image)
    px = image.load()
    with open(elevation_file,'r') as elevation_f:
        for j in range(LINES):
            line = elevation_f.readline().split()
            for i in range(VALUES):
                tc = terrain_costs[ px[i,j][:3] ]
                x = i
                y = j
                e = float(line[i])
                p = Pixel(x,y,e,tc)

                if y not in data:
                    data[y] = dict()
                    data[y][x] = p
                else:
                    data[y][x] = p
    
    #========Actual solution=============

    cost = None # keep track of costs to calculate f.
    solution()
    highlight_checkpoints(trail)
    image.save(output)
