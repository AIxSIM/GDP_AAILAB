from loader.preprocess.mm.fetch_rdnet import fetch_map, build_map
from loader.preprocess.mm.mapmatching import process_gps_and_graph


if __name__ == "__main__":
    
    data_path = "./sets_data/"
    
    # process real
    city = "dj"
    bounds = [127.317, 36.327, 127.371, 36.37]

    map_path = "./sets_data/real/map"

    print("fetching map .... ")
    fetch_map(city, bounds, map_path)
    print("finish!")

    print("building map .... ")
    map_con = build_map(city, map_path)
    print("finish!")
    
    raw_path = "./sets_data/real/raw"
    traj_path = "./sets_data/real/trajectories"
    process_gps_and_graph(city, map_path, data_path, raw_path, traj_path)
    