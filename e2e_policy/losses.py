import torch
import torch.nn.functional as F


def smooth_l1(input, target, beta=1.0):
    return F.smooth_l1_loss(input, target, beta=beta, reduction="none")


def masked_weighted_mean(loss, weight):
    """loss: (B, ...), weight: (B, ...) broadcastable. Weighted mean over valid cells."""
    return (loss * weight).sum() / weight.sum().clamp(min=1.0)


def waypoint_loss(a_hat, a_star, action_valid, action_confidence, lambda_se2=0.1):
    """Masked smooth-L1 waypoint loss + cumulative SE(2) drift term (note sec 8)."""
    valid = action_valid.float()
    conf = action_confidence.float()
    w = valid * conf
    l1 = smooth_l1(a_hat, a_star).mean(dim=-1)          # (B, H)
    L = masked_weighted_mean(l1, w)
    se2 = _cumulative_se2_error(a_hat, a_star, w)
    return L + lambda_se2 * se2, {"L_waypoint": L.item() if not torch.is_grad_enabled() else None}


def _cumulative_se2_error(a_hat, a_star, w):
    def cum(a):
        yaw = torch.atan2(a[..., 2], a[..., 3])          # (B, H)
        psi_prev = torch.cumsum(yaw, dim=1) - yaw
        dx = a[..., 0] * torch.cos(psi_prev) - a[..., 1] * torch.sin(psi_prev)
        dy = a[..., 0] * torch.sin(psi_prev) + a[..., 1] * torch.cos(psi_prev)
        x = torch.cumsum(dx, dim=1)
        y = torch.cumsum(dy, dim=1)
        return torch.stack([x, y], dim=-1)
    err = (cum(a_hat) - cum(a_star)).pow(2).sum(dim=-1).sqrt()
    return masked_weighted_mean(err, w)


def _spatial_grad(x, h, w):
    x = x.reshape(-1, h, w)
    gx = (x[:, :, 1:] - x[:, :, :-1]).abs().mean(dim=(1, 2))
    gy = (x[:, 1:, :] - x[:, :-1, :]).abs().mean(dim=(1, 2)) if h > 1 else torch.zeros_like(gx)
    return gx + gy


def forward_loss(fwd, cur_teacher, future_teacher, grid=(16, 16),
                 depth_target=None, free_target=None, patch_weight=None,
                 target_state_target=None, target_state_valid=None,
                 alpha_depth=0.5, beta_grad=0.1, gamma_free=0.1, delta_state=0.2):
    """Training-only forward dynamics loss (note sec 7/8)."""
    B = cur_teacher.shape[0]
    delta_target = future_teacher - cur_teacher
    pn = F.normalize(fwd["dino_residual"], dim=-1)
    tn = F.normalize(delta_target, dim=-1)
    L_dino = 1.0 - (pn * tn).sum(dim=-1)                 # (B, P)
    w = patch_weight if patch_weight is not None else torch.ones_like(L_dino)
    L = masked_weighted_mean(L_dino, w)
    L_depth = L_grad = L_free = None

    if depth_target is not None:
        L_depth = masked_weighted_mean(smooth_l1(fwd["depth_residual"], depth_target), w)
        h, ww = grid
        gpred = _spatial_grad(fwd["depth_residual"], h, ww)
        gtar = _spatial_grad(depth_target, h, ww)
        L_grad = (gpred - gtar).abs().mean()
        L = L + alpha_depth * L_depth + beta_grad * L_grad

    if free_target is not None:
        L_free = masked_weighted_mean(
            F.binary_cross_entropy_with_logits(fwd["free_logit"], free_target, reduction="none"),
            w,
        )
        L = L + gamma_free * L_free

    if target_state_target is not None:
        st = fwd["target_state"]                          # (B, 3): dx, dy, vis
        pos_t = target_state_target[..., :2]
        vis_t = target_state_target[..., 2:3]
        pos_v = target_state_valid[..., :1] if target_state_valid is not None else torch.ones(B, 1, device=st.device)
        vis_v = target_state_valid[..., 2:3] if target_state_valid is not None else torch.ones(B, 1, device=st.device)
        L_pos = masked_weighted_mean(smooth_l1(st[:, :2], pos_t).mean(-1, keepdim=True), pos_v)
        L_vis = masked_weighted_mean(
            F.binary_cross_entropy_with_logits(st[:, 2:3], vis_t, reduction="none"), vis_v
        )
        L = L + delta_state * (L_pos + L_vis)

    return L, {"L_dino": L_dino.detach().mean().item(), "L_depth": None if L_depth is None else L_depth.item(),
               "L_grad": None if L_grad is None else L_grad.item(),
               "L_free": None if L_free is None else L_free.item()}


def stop_loss(stop_logit, stop_label, task_mask=None):
    """BCE stop loss, masked to tasks with a terminal stop (pointnav/imagegoal)."""
    if stop_label is None:
        return None
    if task_mask is not None and task_mask.sum() > 0:
        sel = task_mask > 0
        return F.binary_cross_entropy_with_logits(stop_logit[sel], stop_label[sel].float())
    return F.binary_cross_entropy_with_logits(stop_logit, stop_label.float())


def build_patch_weight(target_mask, free_mask=None, alpha_target=2.0, beta_free=1.0):
    """patch_weight = 1 + alpha_target * target_mask + beta_free * free_mask (note sec 7.3)."""
    w = torch.ones_like(target_mask)
    w = w + alpha_target * target_mask
    if free_mask is not None:
        w = w + beta_free * free_mask
    return w


def compute_total_loss(out, batch, cfg, lambda_fd=1.0, lambda_inv=0.5, lambda_target=0.5,
                       lambda_stop=0.3, lambda_se2=0.1):
    """Assemble the full training loss (note sec 8)."""
    L_wp, _ = waypoint_loss(
        out["a_hat"], batch["trajectory"],
        batch["action_valid"], batch["action_confidence"], lambda_se2=lambda_se2,
    )

    fd_term = torch.zeros((), device=out["a_hat"].device)
    inv_term = torch.zeros((), device=out["a_hat"].device)
    tgt_term = torch.zeros((), device=out["a_hat"].device)
    stop_term = torch.zeros((), device=out["a_hat"].device)
    logs = {}

    if "forward" in out and out["forward"] is not None:
        pw = batch.get("patch_weight")
        L_fd, lf = forward_loss(
            out["forward"], out["cur_patches_teacher"], out["future_patches_teacher"],
            grid=batch.get("grid", (16, 16)),
            depth_target=batch.get("depth_residual"),
            free_target=batch.get("free_target"),
            patch_weight=pw,
            target_state_target=batch.get("target_state"),
            target_state_valid=batch.get("target_state_valid"),
        )
        fd_term = L_fd
        logs.update(lf)

    if "a_inv" in out and out["a_inv"] is not None:
        L_inv, _ = waypoint_loss(out["a_inv"], batch["trajectory"],
                                 batch["action_valid"], batch["action_confidence"], lambda_se2=0.0)
        inv_term = L_inv

    if "stop_logit" in out and "stop_label" in batch:
        L_stop = stop_loss(out["stop_logit"], batch["stop_label"], batch.get("stop_task_mask"))
        if L_stop is not None:
            stop_term = L_stop

    total = L_wp + lambda_fd * fd_term + lambda_inv * inv_term + lambda_target * tgt_term + lambda_stop * stop_term
    logs["L_waypoint"] = L_wp.item()
    logs["total"] = total.item()
    return total, logs
