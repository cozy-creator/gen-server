# Has two inputs; (1) smapler, (2) scheduler. Outputs a denoise schedule, which
# is represented as an array of floats, called 'sigmas'. This is the rate at which
# noise is removed during the denoising step.
import torch


KSAMPLER_NAMES = ["euler", "euler_ancestral", "heun", "heunpp2","dpm_2", "dpm_2_ancestral",
                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
                  "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm"]
SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]


class DenoiseScheduler:

    NAMESPACE = "denoise_scheduler"

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {
                    "model": ("MODEL",),
                    "scheduler_name": (SCHEDULER_NAMES, ),
                    "sampler_name": (SAMPLER_NAMES, ),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                }
            }
    
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "denoise_scheduler"

    CATEGORY = "sampling"

    def denoise_scheduler(self, model, scheduler_name, sampler_name, steps, denoise=None):
        device = model.load_device
        model = model.model
        if denoise is None or denoise > 0.9999:
            sigmas = self.calculate_sigmas(steps, sampler_name, scheduler_name, model).to(device)
        else:
            new_steps = int(steps/denoise)
            sigmas = self.calculate_sigmas(new_steps, sampler_name, scheduler_name, model).to(device)
            sigmas = sigmas[-(steps + 1):]

        print(f"Sigmas (denoise): {sigmas}")
    
        return ((sigmas, sampler_name, steps), )
    

    def calculate_sigmas(self, steps, sampler_name, scheduler_name, model):
        sigmas = None
        discard_penultimate_sigma = False
        DISCARD_PENULTIMATE_SIGMA_SAMPLERS = set(('dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2'))
        if sampler_name in DISCARD_PENULTIMATE_SIGMA_SAMPLERS:
            steps += 1
            discard_penultimate_sigma = True

        sigmas = comfy.samplers.calculate_sigmas_scheduler(model, scheduler_name, steps)

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
            print(f"Discarding penultimate sigma for {sampler_name}, Sigmas: {sigmas}")

        print(f"Sigmas (new): {sigmas}")

        return sigmas