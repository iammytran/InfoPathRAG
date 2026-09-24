from pathlib import Path
import json
import logging
import os


subquery_with_modality_file = "artifacts/InfoVQA/query_decomposition/test/subqueries_with_modality.json"
old_subquery_with_modality_file = "artifacts/InfoVQA/query_decomposition/test/old_subqueries_with_modality.json"
# logging.basicConfig(level=logging.INFO, 
#                     format='%(asctime)s - %(levelname)s - %(message)s',
#                     filename=os.path.join('debug', 'subquery_with_modality.log'),
#                     filemode='a'
#                     )

def main():
    with open(subquery_with_modality_file, encoding="utf-8") as file:
        content = json.load(file)

    filtered_content = {}
    removed_subqueries = 0
    removed_queries = 0

    for query_id, query in content.items():
        subqueries = query.get("subqueries", [])
        retained_subqueries = [
            subquery for subquery in subqueries if "modality" in subquery
        ]
        removed_subqueries += len(subqueries) - len(retained_subqueries)

        if retained_subqueries:
            query["subqueries"] = retained_subqueries
            filtered_content[query_id] = query
        else:
            removed_queries += 1

    os.replace(subquery_with_modality_file, old_subquery_with_modality_file)
    with open(subquery_with_modality_file, "w", encoding="utf-8") as file:
        json.dump(filtered_content, file, indent=2)
        file.write("\n")

    # logging.info(
    #     "Removed %d subqueries and %d queries without modality",
    #     removed_subqueries,
    #     removed_queries,
    # )

def check_for_missing_modality(subqueries_with_modality_file_path):
    content = ""
    with open(subqueries_with_modality_file_path, 'r') as file:
        content = json.load(file)

    queries_not_have_modality = []
    subqueries_not_have_modality = []
    query_list = []

    # get subquery list
    # get query list
    for query in content.values():
        query_list.append(query)
        subquery_items = query['subqueries']
        subqueries = [sub_q['subquery'] for sub_q in subquery_items]
        # print(subqueries)
        for subquery in subquery_items:
            if 'modality' not in subquery:
                queries_not_have_modality.append(query)
                subqueries_not_have_modality.append(subquery)
    return not len(subqueries_not_have_modality) == 0

if __name__ == "__main__":
    main()