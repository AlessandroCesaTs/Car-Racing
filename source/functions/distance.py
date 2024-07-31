import numpy as np

def distance(s):
    start_x=41
    start_y=71
    size=84
    distance_from_street = None
    distance_from_grass = None
    for distance_x in range(36):
        for distance_y in range(64):
            for dx,dy in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
                point_x,point_y=start_x+(distance_x+5)*dx,start_y+(distance_y+7)*dy
                if 0<=point_x<size and 0<=point_y<size:
                    if 0.39<=s[point_y,point_x]<=0.41:
                    	if distance_from_street is None:
                            distance_from_street = np.sqrt(distance_x**2+distance_y**2)
                    elif distance_from_grass is None:
                        distance_from_grass = np.sqrt(distance_x**2+distance_y**2)
                    if distance_from_street is not None and distance_from_grass is not None:
                        return distance_from_street,distance_from_grass,distance_from_grass-distance_from_street

    if distance_from_street == None:
        distance_from_street=74
    if distance_from_grass == None:
        distance_from_grass=74
    return distance_from_street,distance_from_grass,distance_from_grass-distance_from_street

