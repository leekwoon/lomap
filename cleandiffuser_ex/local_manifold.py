import torch


class LocalManifold:
    def __init__(self, device: torch.device):
        """
        Initializes the LocalManifold class.

        Args:
            device (torch.device): The device on which computations will be performed.
        """
        self.device = device
        self.pc = None
        self.mean_vec = None

    @torch.no_grad()
    def compute_pca(self, manifold_samples: torch.Tensor, tau: float = 0.95):
        """
        Computes the mean vector and principal components for manifold samples.
        Using batched `torch.linalg.eigh` on the Gram matrix X_centered X_centered^T.

        Args:
            manifold_samples (torch.Tensor): Tensor of shape (batch_size, k, d1, d2).
            tau (float, optional): The threshold for cumulative explained variance. Defaults to 0.95.

        Sets:
            self.pc (torch.Tensor): Principal components, shape (batch_size, d, max_pca)
            self.mean_vec (torch.Tensor): Mean vectors, shape (batch_size, 1, d)
        """

        batch_size, k = manifold_samples.shape[0], manifold_samples.shape[1]

        manifold_samples_flat = manifold_samples.reshape(batch_size, k, -1)

        mean_vec = manifold_samples_flat.mean(dim=1, keepdim=True)

        X_centered = manifold_samples_flat - mean_vec

        Cov = torch.bmm(X_centered, X_centered.transpose(2, 1))  # (batch_size, k, k)

        w, V = torch.linalg.eigh(Cov)  # (batch_size, k), (batch_size, k, k)

        idx = torch.argsort(w, dim=1, descending=True)  # (batch_size, k)
        w_sorted = torch.gather(w, 1, idx)              # (batch_size, k)
        
        V_sorted = V[torch.arange(batch_size).unsqueeze(-1), :, idx]

        total_var = w_sorted.sum(dim=1, keepdim=True)  # (batch_size, 1)
        var_explained = w_sorted / (total_var + 1e-8)  # (batch_size, k)
        cum_var_explained = torch.cumsum(var_explained, dim=1)  # (batch_size, k)

        threshold_tensor = torch.full((batch_size, 1), tau, device=self.device)
        pca_components = torch.sum(cum_var_explained < threshold_tensor, dim=1) + 1  # (batch_size,)
        max_pca = pca_components.max().item()
        w_top = w_sorted[:, :max_pca]            # (batch_size, max_pca)        
        V_top = V_sorted[:, :max_pca, :]
        V_top = V_top.transpose(2, 1)
        
        w_top_sqrt_inv = 1.0 / (w_top.clamp_min(1e-8).sqrt())  # (batch_size, max_pca)

        pc_raw = torch.bmm(X_centered.transpose(1,2), V_top)  # (batch_size, d, max_pca)
        pc = pc_raw * w_top_sqrt_inv.unsqueeze(1)             # (batch_size, d, max_pca)

        self.mean_vec = mean_vec  # (batch_size, 1, d)
        self.pc = pc              # (batch_size, d, max_pca)

    @torch.no_grad()
    def project_points(self, points: torch.Tensor):
        """
        Projects a batch of points onto their respective PCA subspaces.

        Args:
            points (torch.Tensor): Batch of points to be projected, shape (batch_size, dimension).
            pc (torch.Tensor): Principal components for each batch, shape (batch_size, dimension, max_pca).
            mean_vec (torch.Tensor): Mean vectors for each batch, shape (batch_size, 1, dimension).

        Returns:
            torch.Tensor: The projected points, shape (batch_size, dimension).
        """
        batch_size, dims = points.shape[0], points.shape[1:]
        max_pca = self.pc.shape[2]

        points_flat = points.reshape(batch_size, -1)

        # Center the points
        x_centered = points_flat - self.mean_vec.squeeze(1)  # Shape: (batch_size, dimension)

        # Project the centered points onto the principal components
        x_pca_coords = torch.bmm(x_centered.unsqueeze(1), self.pc)  # Shape: (batch_size, 1, max_pca)
        x_new = self.mean_vec.squeeze(1) + torch.bmm(x_pca_coords, self.pc.transpose(1, 2)).squeeze(1)  # Shape: (batch_size, dimension)

        x_new = x_new.reshape(batch_size, *dims)

        return x_new

    @torch.no_grad()
    def project_gradients(self, grads: torch.Tensor):
        """
        Projects a batch of gradients onto their respective PCA subspaces.

        Args:
            grads (torch.Tensor): Gradients to be projected, shape (batch_size, dimension).
            pc (torch.Tensor): Principal components for each batch, shape (batch_size, dimension, max_pca).

        Returns:
            torch.Tensor: The projected gradients, shape (batch_size, dimension).
        """
        batch_size, dims = grads.shape[0], grads.shape[1:]

        grads_flat = grads.reshape(batch_size, - 1)

        # Compute the projection: proj_grad = pc @ (pc^T @ grad)
        intermediate = torch.bmm(self.pc.transpose(1, 2), grads_flat.unsqueeze(2))  # Shape: (batch_size, max_pca, 1)
        projected_grads = torch.bmm(self.pc, intermediate).squeeze(2)  # Shape: (batch_size, dimension)

        projected_grads = projected_grads.reshape(batch_size, *dims)

        return projected_grads
    