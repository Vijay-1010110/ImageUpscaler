from .sr_residual import ResidualSR


def build_model(config):
    if config["model_name"] == "residual_sr":
        return ResidualSR(
            scale=config["scale"],
            num_res_blocks=config["num_res_blocks"]
        )
    else:
        raise ValueError(f"Unknown model: {config['model_name']}")
