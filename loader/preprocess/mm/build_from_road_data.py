import matplotlib.pyplot as plt
plt.switch_backend("agg")
import os
from os.path import join
from leuvenmapmatching.map.inmem import InMemMap
import osmnx as ox
from typing import List
import xml.etree.ElementTree as ET
import pandas as pd


def fetch_map(city: str, bounds: List[float], save_path: str):
    if os.path.exists(join(save_path, f"{city}.graphml")):
        return
    north, south, east, west = bounds[3], bounds[1], bounds[2], bounds[0]
    g = ox.graph_from_bbox(north, south, east, west, network_type='drive')
    ox.save_graphml(g, join(save_path, f"{city}.graphml"))

def parse_nodes_to_dataframe(file_path, tag):
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = []
    for node in root.findall(tag):
        data.append(node.attrib)

    df = pd.DataFrame(data)
    return df

# build map
def build_map_from_road_data(city: str, map_path: str, road_path: str, add_reverse=True):
    print("build_map start!")

    xml_file_node = join(road_path, "node.xml")
    df_node = parse_nodes_to_dataframe(xml_file_node, ".//node")

    xml_file_edge = join(road_path, "edge.xml")
    df_edge = parse_nodes_to_dataframe(xml_file_edge, ".//edge")

    plt.clf()
    map_con = InMemMap(name=f"map_{city}", use_latlon=True, use_rtree=True, index_edges=True, dir=map_path)

    # construct road network

    # nid_to_cmpct: dict, key: unique node id, value: compact node id
    # cmpct_to_nid: list, unique node id (ordered by compact node id)
    # row['y']: float, latitude
    # row['x']: float, longitude
    nid_to_cmpct = dict()
    cmpct_to_nid = []
    for index, row in df_node.iterrows():
        try:
            node_id = int(row['id'])
        except:
            continue
        if node_id not in nid_to_cmpct:
            nid_to_cmpct[node_id] = len(cmpct_to_nid)
            cmpct_to_nid.append(node_id)
        cid = nid_to_cmpct[node_id]
        map_con.add_node(cid, (float(row['y']), float(row['x'])))

    for index, row in df_edge.iterrows():
        # edge_id = row['id']
        try:
            node_id_1 = int(row['from'])
            node_id_2 = int(row['to'])
        except:
            continue

        if node_id_1 not in nid_to_cmpct:
            nid_to_cmpct[node_id_1] = len(cmpct_to_nid)
            cmpct_to_nid.append(node_id_1)
        if node_id_2 not in nid_to_cmpct:
            nid_to_cmpct[node_id_2] = len(cmpct_to_nid)
            cmpct_to_nid.append(node_id_2)
        cid1 = nid_to_cmpct[node_id_1]
        cid2 = nid_to_cmpct[node_id_2]
        print(cid1, cid2)
        import pdb
        pdb.set_trace()
        map_con.add_edge(cid1, cid2)
        if add_reverse:
            map_con.add_edge(cid2, cid1)
    map_con.dump()
    return map_con


if __name__ == "__main__":
    city = "djr"
    bounds = [127.317, 36.327, 127.371, 36.37]
    map_path = "./sets_data/real_from_road_data/map"
    road_path = "/home/aailab/data4/wp03052/synthesis/Daejeon-Network"

    print("fetching map .... ")
    fetch_map(city, bounds, map_path)
    print("finish!")

    print("building map .... ")
    map_con = build_map_from_road_data(city, map_path, road_path)
    print("finish!")

