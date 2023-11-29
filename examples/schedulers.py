import torch.optim.lr_scheduler as lr_scheduler

def create_scheduler(optimizer_name, scheduler_type, max_steps, lr):
    if scheduler_type == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer_name, step_size=1000, gamma=0.847
        )
    elif scheduler_type == "cosineannealing":
        scheduler = lr_scheduler.ChainedScheduler(
            [
                lr_scheduler.CosineAnnealingLR(
                    optimizer_name,
                    T_max=max_steps,
                    eta_min=lr / 10
                )])
    elif scheduler_type == "chain":
        scheduler = lr_scheduler.ChainedScheduler(
            [
                lr_scheduler.LinearLR(
                    optimizer_name, start_factor=0.01, total_iters=100
                ),
                lr_scheduler.MultiStepLR(
                    optimizer_name,
                    milestones=[
                        max_steps // 2,
                        max_steps * 3 // 4,
                        max_steps * 9 // 10,
                    ],
                    gamma=0.33,
                ),
            ]
        )
    elif scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Invalid scheduler type: {scheduler_type}")

    return scheduler