# demo/trainable_uq.py
# Trainable UQ with Equinox: model maps (x+w, Σ) -> (μ_out, Λ_out)
# Loss: -log N(y; μ_out, Λ_out^{-1}) with w ~ N(0, Σ), Σ ~ Inv-Wishart(ν, Ψ)

import math
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


def true_function(x: jnp.ndarray) -> jnp.ndarray:
    """Ground-truth mapping R^d -> R^k for synthetic data."""
    # Assumes d >= 3, k == 2; adjust for other dims accordingly.
    y1 = jnp.sin(x[0]) + x[1] ** 2
    y2 = x[2] + 0.5 * x[0] * x[1]
    return jnp.array([y1, y2])


v_true_function = jax.vmap(true_function)


def sample_wishart(key: jax.Array, df: int, S: jnp.ndarray) -> jnp.ndarray:
    """Wishart(df, S) via sum-of-outer-products (requires integer df)."""
    p = S.shape[0]
    Ls = jnp.linalg.cholesky(S + 1e-8 * jnp.eye(p))
    keys = jax.random.split(key, df)
    eps = jax.vmap(lambda k: jax.random.normal(k, (p,)))(keys)  # (df, p)
    Z = eps @ Ls.T  # (df, p) with rows ~ N(0, S)
    W = Z.T @ Z  # (p, p)
    return W


def sample_inverse_wishart(key: jax.Array, df: int, Psi: jnp.ndarray) -> jnp.ndarray:
    """Inverse-Wishart(df, Psi) using W ~ Wishart(df, Psi^{-1}), Σ = W^{-1}."""
    S = jnp.linalg.inv(Psi)
    W = sample_wishart(key, df, S)
    # Stabilize inversion with a small jitter.
    Wj = W + 1e-6 * jnp.eye(W.shape[0])
    Sigma = jnp.linalg.inv(Wj)
    return Sigma


def sample_w(key: jax.Array, Sigma: jnp.ndarray) -> jnp.ndarray:
    """Sample w ~ N(0, Sigma)."""
    p = Sigma.shape[0]
    Ls = jnp.linalg.cholesky(Sigma + 1e-8 * jnp.eye(p))
    z = jax.random.normal(key, (p,))
    return Ls @ z


def _lower_cholesky_from_vector(
    diag_raw: jnp.ndarray, off_raw: jnp.ndarray, k: int, jitter: float = 1e-5
) -> jnp.ndarray:
    """Build lower-triangular Cholesky L from raw vectors. diag via softplus."""
    diag = jax.nn.softplus(diag_raw) + jitter  # ensure strictly positive
    L = jnp.zeros((k, k))
    L = L.at[jnp.diag_indices(k)].set(diag)
    if k > 1:
        rows, cols = jnp.tril_indices(k, -1)
        L = L.at[rows, cols].set(off_raw)
    return L


class UQNet(eqx.Module):
    mean_net: eqx.nn.MLP
    prec_net: eqx.nn.MLP
    in_dim: int
    out_dim: int
    chol_jitter: float

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 64,
        depth: int = 2,
        chol_jitter: float = 1e-5,
        key: jax.Array = jax.random.PRNGKey(0),
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.chol_jitter = chol_jitter

        k1, k2 = jax.random.split(key, 2)
        # Mean head
        self.mean_net = eqx.nn.MLP(
            in_size=in_dim, out_size=out_dim, width_size=width, depth=depth, key=k1
        )
        # Precision head outputs diag (k) + strictly lower-tri (k*(k-1)//2)
        p_out = out_dim + (out_dim * (out_dim - 1)) // 2
        self.prec_net = eqx.nn.MLP(
            in_size=in_dim, out_size=p_out, width_size=width, depth=depth, key=k2
        )

    def _forward_one(
        self, xw: jnp.ndarray, Sigma: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Single-sample forward: returns (mu_out, L) with Λ = L L^T."""
        features = jnp.concatenate([xw, Sigma.reshape(-1)], axis=0)
        mu = self.mean_net(features)  # (k,)
        pvec = self.prec_net(features)  # (k + k*(k-1)/2,)
        k = self.out_dim
        diag_raw = pvec[:k]
        off_raw = pvec[k:]
        L = _lower_cholesky_from_vector(diag_raw, off_raw, k, jitter=self.chol_jitter)
        return mu, L

    def __call__(
        self, xw: jnp.ndarray, Sigma: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Batched or single forward.
        - xw: (d,) or (B, d)
        - Sigma: (d, d) or (B, d, d)
        Returns:
          - mu: (k,) or (B, k)
          - L:  (k, k) or (B, k, k), precision Λ = L L^T
        """
        if xw.ndim == 1:
            return self._forward_one(xw, Sigma)
        else:
            return jax.vmap(self._forward_one, in_axes=(0, 0))(xw, Sigma)


def nll_gaussian_with_precision(
    y: jnp.ndarray, mu: jnp.ndarray, L: jnp.ndarray
) -> jnp.ndarray:
    """
    Negative log-likelihood for y ~ N(mu, Λ^{-1}) with Λ = L L^T.
    Supports shapes:
      - y: (k,), mu: (k,), L: (k, k)
      - y: (B, k), mu: (B, k), L: (B, k, k)
    """
    k = y.shape[-1]
    if y.ndim == 1:
        e = y - mu  # (k,)
        # logdet(Λ) = 2 * sum(log(diag(L)))
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        q = jnp.sum((e @ L) ** 2)  # ||e L||^2 = e^T Λ e
        return 0.5 * (k * jnp.log(2 * jnp.pi) - logdet + q)
    else:
        e = y - mu  # (B, k)
        diagL = jnp.diagonal(L, axis1=-2, axis2=-1)  # (B, k)
        logdet = 2.0 * jnp.sum(jnp.log(diagL), axis=-1)  # (B,)
        eL = jnp.einsum("bk,bkj->bj", e, L)  # (B, k)
        q = jnp.sum(eL**2, axis=-1)  # (B,)
        return 0.5 * (k * jnp.log(2 * jnp.pi) - logdet + q)


def make_loss_fn(
    model: UQNet,
    df_iw: int,
    Psi: jnp.ndarray,
    key: jax.Array,
):
    """Create a loss function closure with its own RNG splitting for reproducibility."""

    def loss_fn(
        model, x_batch: jnp.ndarray, y_batch: jnp.ndarray, key: jax.Array
    ) -> jnp.ndarray:
        B, d = x_batch.shape
        k = y_batch.shape[-1]
        # Sample Σ and w for each sample in the batch
        subkeys = jax.random.split(key, 2)
        k_sigma, k_w = subkeys

        sigma_keys = jax.random.split(k_sigma, B)
        w_keys = jax.random.split(k_w, B)

        sample_sigma = jax.vmap(lambda kk: sample_inverse_wishart(kk, df_iw, Psi))
        Sigmas = sample_sigma(sigma_keys)  # (B, d, d)

        sample_w_v = jax.vmap(sample_w, in_axes=(0, 0))
        Ws = sample_w_v(w_keys, Sigmas)  # (B, d)

        Xw = x_batch + Ws  # (B, d)

        # Forward through model
        mu_pred, L_pred = model(Xw, Sigmas)  # (B, k), (B, k, k)

        # NLL
        nll = nll_gaussian_with_precision(y_batch, mu_pred, L_pred)  # (B,)
        return jnp.mean(nll)

    return loss_fn


def main():
    # Configs
    key = jax.random.PRNGKey(0)
    d = 3  # input dimension
    k = 2  # output dimension
    n_total = 4096
    train_frac = 0.8
    n_train = int(n_total * train_frac)
    n_val = n_total - n_train

    noise_y_std = 0.1

    # Inverse-Wishart Σ ~ IW(ν, Ψ)
    df_iw = int(d + 5)  # integer df; must be > d - 1
    Psi_scale = 0.2
    Psi = (Psi_scale**2) * jnp.eye(d)

    # Model config
    width = 64
    depth = 2
    lr = 1e-3
    steps = 3000
    batch_size = 128
    chol_jitter = 1e-5

    # Generate synthetic dataset
    key, kx, ky = jax.random.split(key, 3)
    X = jax.random.normal(kx, (n_total, d))
    Y_clean = v_true_function(X)
    Y = Y_clean + noise_y_std * jax.random.normal(ky, (n_total, k))

    X_train, X_val = X[:n_train], X[n_train:]
    Y_train, Y_val = Y[:n_train], Y[n_train:]

    # Build model
    in_dim = d + d * d
    key, k_model = jax.random.split(key)
    model = UQNet(
        in_dim=in_dim,
        out_dim=k,
        width=width,
        depth=depth,
        chol_jitter=chol_jitter,
        key=k_model,
    )

    # Optimizer
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    # Loss function
    loss_fn = make_loss_fn(model, df_iw=df_iw, Psi=Psi, key=key)

    @eqx.filter_value_and_grad
    def training_loss(model, xb, yb, key):
        return loss_fn(model, xb, yb, key)

    @eqx.filter_jit
    def train_step(model, opt_state, xb, yb, key):
        loss, grads = training_loss(model, xb, yb, key)
        updates, opt_state = opt.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @eqx.filter_jit
    def eval_step(model, xb, yb, key):
        return loss_fn(model, xb, yb, key)

    # Training loop
    num_batches = math.ceil(n_train / batch_size)
    for step in range(1, steps + 1):
        key, kperm, kstep = jax.random.split(key, 3)
        perm = jax.random.permutation(kperm, n_train, independent=True)
        Xb = X_train[perm]
        Yb = Y_train[perm]

        # One epoch over shuffled data
        epoch_loss = 0.0
        for b in range(num_batches):
            s = b * batch_size
            e = min(n_train, (b + 1) * batch_size)
            xb = Xb[s:e]
            yb = Yb[s:e]
            key, kbatch = jax.random.split(key)
            model, opt_state, loss = train_step(model, opt_state, xb, yb, kbatch)
            epoch_loss += float(loss) * (e - s)
        epoch_loss /= n_train

        # Periodic validation
        if step % 100 == 0 or step == 1:
            key, kval = jax.random.split(key)
            val_loss = float(eval_step(model, X_val, Y_val, kval))
            print(
                f"Step {step:5d} | train NLL: {epoch_loss:.4f} | val NLL: {val_loss:.4f}"
            )

    # Final eval
    key, kfinal = jax.random.split(key)
    final_val = float(eval_step(model, X_val, Y_val, kfinal))
    print(f"Final validation NLL: {final_val:.4f}")


if __name__ == "__main__":
    main()
