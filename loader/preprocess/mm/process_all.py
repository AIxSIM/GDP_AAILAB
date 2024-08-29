from loader.preprocess.mm.fetch_rdnet import fetch_map, build_map
from loader.preprocess.mm.mapmatching import process_gps_and_graph


if __name__ == "__main__":
    
    data_path = "./sets_data/"
    
    # process real2
    city = "jakarta"
    bounds = [106.833, -6.21, 106.970, -6.091]

    map_path = "./sets_data/real2/map"

    print("fetching map .... ")
    fetch_map(city, bounds, map_path)
    print("finish!")

    print("building map .... ")
    map_con = build_map(city, map_path)
    print("finish!")
    
    raw_path = "./sets_data/real2/raw"
    traj_path = "./sets_data/real2/trajectories"
    process_gps_and_graph(city, map_path, data_path, raw_path, traj_path)
    