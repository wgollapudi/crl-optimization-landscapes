# landscape/metric_keys.py

LOSS_KEYS = {
    "loss",
    "recon_img",
    "kl",
    "recon_anchor",
}

AUX_LOSS_PREFIXES = (
    "mmd",
    "graph",
    "sparsity",
)

DIAGNOSTIC_PREFIXES = (
    "mu_",
    "logvar_",
)


def is_loss_key(key: str) -> bool:
    if key in LOSS_KEYS:
        return True
    return key.startswith(AUX_LOSS_PREFIXES)


def is_diagnostic_key(key: str) -> bool:
    return key.startswith(DIAGNOSTIC_PREFIXES)


def split_metrics(metrics: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    loss_metrics = {
        k: v for k, v in metrics.items()
        if is_loss_key(k)
    }
    diagnostics = {
        k: v for k, v in metrics.items()
        if not is_loss_key(k)
    }
    return loss_metrics, diagnostics
