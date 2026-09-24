# Khai báo các target không tạo ra file thực tế (chỉ là tên câu lệnh)
.PHONY: all install preprocessing decompose_query embed retrieve visualize_result run_lilac

# Lệnh mặc định khi gõ "make" không truyền tham số
all: run_lilac

install:
	cd LILaC
	conda create -n mmembed python=3.10 -y && conda activate mmembed && pip install -r conda_environments/mmembed.txt
	./models/download_embedders.sh           # MM-Embed, UniME, mmE5
	conda env update --file conda_environments/lilac-qwen.yaml
	./models/download_generator.sh           # Qwen2.5-VL-7B + Qwen2.5-72B-Instruct

	git clone https://github.com/Dao-AILab/flash-attention.git
	git checkout v2.2.0
	MAX_JOBS=4 python setup.py install


preprocessing:
	cd ..
	./models/download_layout_analyzers.sh --only mineru         # one-time (installs CLI + weights)

	python src/lilac/lcg_constructor/preprocessing/mineru_to_lilac.py --layout artifacts/InfoVQA/layout_mineru/after_process_tiling --images datasets/InfoVQA/image_components/test --output datasets/InfoVQA/parsed_documents/test --crops_out artifacts/InfoVQA/crops_out/after_process_tiling --num_gpus 4


decompose_query:
	@bash -c 'eval "$$(conda shell.bash hook)" && conda activate lilac-qwen
	./scripts/query_decomposition/query_decomposer.sh 
	./scripts/query_decomposition/modality_estimator.sh
	
embed: 
	@echo "=== [1/3] Embedding InfoVQA ==="
	@bash -c 'eval "$$(conda shell.bash hook)" && conda activate mmembed && ./scripts/parse_multimodal_document/step8_embed_mmembed.sh -b "InfoVQA"'
	@bash -c 'eval "$$(conda shell.bash hook)" && conda activate mmembed && ./scripts/query_decomposition/queryset_embedder.sh --embedder mmembed

retrieve: install preprocessing decompose_query embed
	@echo "=== [2/3] Retrieving ==="
	@bash -c 'eval "$$(conda shell.bash hook)" && conda activate mmembed && CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/experiments/retriever_accuracy.sh -e "MM-Embed" -b "InfoVQA"'

visualize_result: embed retrieve
	@echo "=== [3/3] Visualizing result ==="
	./scripts/experiment_visualization/retriever_accuracy.sh 

run_lilac: visualize_result
	@echo "✅ ALL STEPS COMPLETED SUCCESSFULLY!"
