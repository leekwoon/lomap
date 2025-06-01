import os
import faiss
import numpy as np


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

def apply_dim_weights(data: np.ndarray, dim_weights: np.ndarray) -> np.ndarray:
    if dim_weights is None:
        return data
    if dim_weights.shape[0] != data.shape[1]:
        raise ValueError("dim_weights must match data.shape[1].")
    return data * dim_weights


class FaissIndexIVFWrapper:

    def __init__(
        self,
        similarity_metric: str = "l2",
        nlist: int = 1000,
        data = None,
        dim_weights: np.ndarray = None, 
        device: str = "cuda:0",
    ):
        self.num_data, *self.dims = data.shape
        self.dim_flat = int(np.prod(self.dims))

        self.similarity_metric = similarity_metric.lower()
        self.nlist = nlist
        self.nprobe = 5 # max(1, self.nlist // 10)
        self.data_flat = data.reshape(self.num_data, -1).copy(order = 'C')
        self.device = device
        self.dim_weights_flat = dim_weights.reshape(-1) if dim_weights is not None else None

        if device.lower() == "cpu":
            raise ValueError("[FaissIndexIVFPQWrapper] do not support cpu device")
        elif device.lower().startswith("cuda:"):
            try:
                gpu_id_str = device.split(":")[1]
                self.gpu_id = int(gpu_id_str)
            except (IndexError, ValueError):
                raise ValueError(f"wrong device: {device}")
        else:
            raise ValueError(f"wrong device: {device}")

        # GPU resource
        self.gpu_res = faiss.StandardGpuResources()

        # index
        self.index_cpu = None
        self.index = None
        self.size = 0

        self.build_index(self.data_flat)

    def build_index(self, data: np.ndarray):
        if data.ndim != 2 or data.shape[1] != self.dim_flat:
            raise ValueError(f"data.shape={data.shape}, expected (N, {self.dim_flat})")

        # 1) scale if needed
        data_scaled = apply_dim_weights(data, self.dim_weights_flat)

        # 2) metric
        if self.similarity_metric == 'cosine':
            data_scaled = normalize_vectors(data_scaled)
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            metric = faiss.METRIC_L2

        n_data = data_scaled.shape[0]
        print(f"[FaissIndexIVFFlatWrapper] building IVF-Flat with {n_data} vectors...")

        # 3) quantizer
        if metric == faiss.METRIC_INNER_PRODUCT:
            quantizer = faiss.IndexFlatIP(self.dim_flat)
        else:
            print(self.dim_flat)
            quantizer = faiss.IndexFlatL2(self.dim_flat)

        # 4) create IVF Flat
        index_ivf = faiss.IndexIVFFlat(quantizer, self.dim_flat, self.nlist, metric)
        self.index_cpu = index_ivf

        # train
        if not index_ivf.is_trained:
            print("[FaissIndexIVFFlatWrapper] Training IVF quantizer...")
            index_ivf.train(data_scaled)
            print("[FaissIndexIVFFlatWrapper] Training done.")

        # add
        index_ivf.add(data_scaled)
        self.size = n_data
        print(f"[FaissIndexIVFFlatWrapper] CPU IVF-Flat index has {self.size} vectors")

        # to GPU
        self.index = faiss.index_cpu_to_gpu(self.gpu_res, self.gpu_id, index_ivf)
        self.index.nprobe = self.nprobe
        print(f"[FaissIndexIVFFlatWrapper] Moved IVF-Flat index to GPU. nprobe={self.index.nprobe}")

    def search(self, queries: np.ndarray, k: int):
        queries_flat = queries.reshape(-1, self.dim_flat)

        if self.index is None:
            raise RuntimeError("Index not built or loaded yet.")

        if queries_flat.shape[1] != self.dim_flat:
            raise ValueError(f"queries_flat shape {queries_flat.shape}, expected (B, {self.dim_flat})")

        # scale queries
        queries_flat_scaled = apply_dim_weights(queries_flat, self.dim_weights_flat)

        # cos => normalize
        if self.similarity_metric == 'cosine':
            queries_flat_scaled = normalize_vectors(queries_flat_scaled)

        distances, indices = self.index.search(queries_flat_scaled, k)
        # these distances are the actual (approx for IP? Actually IVFFlat => exact, but L2 => exact)
        return distances, indices

    def get_original_vectors(self, indices: np.ndarray):
        flat_indices = indices.ravel()
        out_flat = self.data_flat[flat_indices]
        B, K = indices.shape
        out = out_flat.reshape(B, K, *self.dims)
        return out





