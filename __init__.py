import torch


class LatentClamp:

	@classmethod
	def INPUT_TYPES(s):
		return {
		    "required": {
		        "samples": ("LATENT", ),
		        "min": ("FLOAT", {
		            "default": -1.0,
		            "min": -10.0,
		            "max": 10.0,
		            "step": 0.01,
		            "tooltip": "Values in `samples` below this value will be multiplied by `outside_multiplier`."
		        }),
		        "max": ("FLOAT", {
		            "default": 1.0,
		            "min": -10.0,
		            "max": 10.0,
		            "step": 0.01,
		            "tooltip": "Values in `samples` above this value will be multiplied by `outside_multiplier`."
		        }),
		        "inside_multiplier": ("FLOAT", {
		            "default": 1.0,
		            "min": 0.0,
		            "max": 10.0,
		            "step": 0.1,
		            "tooltip": "Values in `samples` between `min` and `max` will be multiplied by this value."
		        }),
		        "outside_multiplier": ("FLOAT", {
		            "default": 0.5,
		            "min": 0.0,
		            "max": 10.0,
		            "step": 0.1,
		            "tooltip": "Values in `samples` outside of `min` and `max` will be multiplied by this value."
		        }),
		        "extra_noise": ("FLOAT", {
		            "default": 0.0,
		            "min": 0.0,
		            "max": 100.0,
		            "step": 0.1,
		            "tooltip": "Add compensatory noise to the values in `samples`."
		        }),
		    }
		}

	RETURN_TYPES = ("LATENT", )
	FUNCTION = "op"
	CATEGORY = "latent/advanced"

	def op(self, samples, min, max, inside_multiplier, outside_multiplier, extra_noise):
		samples_out = samples.copy()
		s1 = samples["samples"]
		mask_min = s1 < min
		mask_max = s1 > max
		mask_inside = ~mask_min & ~mask_max

		if extra_noise:
			outside_noise = torch.randn_like(s1) * extra_noise * (1 - outside_multiplier)
			inside_noise = torch.randn_like(s1) * extra_noise * (1 - inside_multiplier)
		else:
			inside_noise = 0
			outside_noise = 0

		s1 = torch.where(mask_min, s1 * outside_multiplier + outside_noise, s1)
		s1 = torch.where(mask_max, s1 * outside_multiplier + outside_noise, s1)
		s1 = torch.where(mask_inside, s1 * inside_multiplier + inside_noise, s1)
		samples_out["samples"] = s1
		return (samples_out, )


NODE_CLASS_MAPPINGS = {
    "LatentClamp": LatentClamp,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentClamp": "Latent Clamp",
}
