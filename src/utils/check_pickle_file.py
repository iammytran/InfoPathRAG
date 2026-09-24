import pandas as pd
import pickle
import logging
import os
import json

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=os.path.join('debug', 'pickle.log'),
                    filemode='a')

output_file = "debug/pickle.json"

pickle_file = "artifacts/InfoVQA/components/graph.pickle"

def custom_serializer(obj):
    if hasattr(obj, "component_id"):
        return f"{obj.component_id}"
    return str(obj)

def main():
    # df = pd.read_pickle("artifacts/InfoVQA/components/graph.pickle")
    # with open(pickle_file, 'rb') as pickle_file:
    #     data = pickle.load(pickle_file)
    
    with open(pickle_file, "rb") as file:
        G = pickle.load(file)

    data = []
    # print(type(G))
    with open(output_file, 'w') as file:
        data = G.get_intra_document_edges_dict()
        json.dump(G.get_intra_document_edges_dict(), file, indent=2, default=custom_serializer)

    count = len(data)
    print(count)
    
    # logging.info(G.)
    
    # print(f"Đang xử lý đồ thị có: {G.number_of_nodes()} nodes và {G.number_of_edges()} edges.")
    
    # # 2. Tạo khung vẽ (Figure)
    # plt.figure(figsize=(12, 12))  # Kích thước ảnh (12x12 inches)
    
    # # 3. Tính toán thuật toán sắp xếp vị trí các nút (Layout)
    # # spring_layout giúp các nút tự động giãn cách đều nhau trông như một mạng lưới sinh học
    # pos = nx.spring_layout(G, k=0.15, seed=42) 
    
    # # 4. Vẽ các thành phần của đồ thị
    # nx.draw_networkx_nodes(G, pos, node_size=100, node_color="skyblue", alpha=0.8)
    # nx.draw_networkx_edges(G, pos, width=0.5, edge_color="gray", alpha=0.5)
    
    # # Nếu đồ thị nhỏ và bạn muốn hiện nhãn (labels) tên của từng nút:
    # # nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
    
    # plt.title("Trực quan hóa Đồ thị từ file Pickle", fontsize=14)
    # plt.axis("off")  # Tắt trục tọa độ X-Y đi cho đẹp
    
    # # 5. Lưu đồ thị thành file ảnh trên server
    # output_image = "workspace_graph.png"
    # plt.savefig(output_image, bbox_inches="tight", dpi=300)
    # plt.close()
    
    # print(f"Đã vẽ xong! File ảnh được lưu tại: {output_image}")
    return

if __name__=="__main__":
    main()