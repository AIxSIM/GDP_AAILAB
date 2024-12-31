from loader.preprocess.mm.build_from_road_data import fetch_map, build_map_from_road_data
from loader.preprocess.mm.mapmatching import process_gps_and_graph


if __name__ == "__main__":
    
    data_path = "./sets_data/"
    
    # process real
    city = "djr"
    bounds = [127.317, 36.327, 127.371, 36.37]

    prefix = "./sets_data/real_from_road_data"
    map_path = f"{prefix}/map"
    road_path = "/home/aailab/data4/wp03052/synthesis/Daejeon-Network"

    print("fetching map .... ")
    fetch_map(city, bounds, map_path)
    print("finish!")

    print("building map .... ")
    map_con = build_map_from_road_data(city, map_path, road_path)
    print("finish!")
    
    raw_path = f"{prefix}/raw"
    traj_path = f"{prefix}/trajectories"
    process_gps_and_graph(city, map_path, data_path, raw_path, traj_path)
