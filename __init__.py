import os
import folder_paths
import numpy as np
import torch
from comfy.utils import ProgressBar
from .trt_utilities import Engine
from .utilities import download_file, ColoredLogger
import comfy.model_management as mm
import time
import tensorrt

logger = ColoredLogger("ComfyUI-Upscaler-TensorRT-Advanced")


class UpscalerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Images to be upscaled. Resolution must be between 256 and 1280 px if no build options has been specified."}
                ),
                "engine": (
                    "UPSCALER_TRT_ENGINE_PKG",
                    {"tooltip": "TensorRT engine built and loaded"}
                ),
            },
            "optional": {
                "resize": (
                    "UPSCALER_TRT_RESIZE",
                    {"tooltip": "Resize the image after the x4 model upscale"}
                ),
            }
        }
    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscaler_tensorrt"
    CATEGORY = "TensorRT/Upscaler"
    DESCRIPTION = "Upscale images with TensorRT"

    def upscaler_tensorrt(self, image, engine, resize=None):
        options = engine["options"]
        engine = engine["engine"]

        images_bchw = image.permute(0, 3, 1, 2)
        B, C, H, W = images_bchw.shape

        if W < options.width_min:
            raise ValueError(f"The input image width ({W}) is lower than the TensorRT engine minimum width ({options.width_min}). Please set options to the loader accordingly.")
        if W > options.width_max:
            raise ValueError(f"The input image width ({W}) is greater than the TensorRT engine maximum width ({options.width_max}). Please set options to the loader accordingly.")
        if H < options.height_min:
            raise ValueError(f"The input image height ({H}) is lower than the TensorRT engine minimum height ({options.height_min}). Please set options to the loader accordingly.")
        if H > options.height_max:
            raise ValueError(f"The input image height ({H}) is greater than the TensorRT engine maximum height ({options.height_max}). Please set options to the loader accordingly.")
        if B < options.batch_min:
            raise ValueError(f"The input batch size ({B}) is lower than the TensorRT engine minimum batch ({options.batch_min}). Please set options to the loader accordingly.")
        if B > options.batch_max:
            raise ValueError(f"The input batch size ({B}) is greater than the TensorRT engine maximum batch ({options.batch_max}). Please set options to the loader accordingly.")

        if resize is None:
            final_width, final_height = W*4, H*4
        else:
            final_width, final_height = resize["width"], resize["height"]
        logger.info(f"Upscaling {B} images from H:{H}, W:{W} to H:{H*4}, W:{W*4} | Final resolution: H:{final_height}, W:{final_width}")

        shape_dict = {
            "input": {"shape": (1, 3, H, W)},
            "output": {"shape": (1, 3, H*4, W*4)},
        }
        # setup engine
        engine.activate()
        engine.allocate_buffers(shape_dict=shape_dict)

        cudaStream = torch.cuda.current_stream().cuda_stream
        pbar = ProgressBar(B)
        images_list = list(torch.split(images_bchw, split_size_or_sections=1))

        upscaled_frames = torch.empty((B, C, final_height, final_width), dtype=torch.float32, device=mm.intermediate_device()) # offloaded to cpu

        for i, img in enumerate(images_list):
            result = engine.infer({"input": img}, cudaStream)
            result = result["output"]
            if W*4 != final_width or H*4 != final_height:
                # must resize
                if W*4 > final_width and H*4 > final_height:
                    # downscale, let's use the specialized area mode
                    mode='area'
                    antialias=False # not compatible/needed for downscaling
                else:
                    # upscale, let's use bicubic
                    mode='bicubic'
                    antialias=True # needed for quality upscaling
                result = torch.nn.functional.interpolate(
                    result,
                    size=(final_height, final_width),
                    mode=mode,
                    antialias=antialias
                )
            upscaled_frames[i] = result.to(mm.intermediate_device())
            pbar.update(1)

        output = upscaled_frames.permute(0, 2, 3, 1)
        engine.reset() # frees engine vram
        mm.soft_empty_cache()

        logger.info(f"Output shape: {output.shape}")
        return (output,)

class ResizeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 2560, "min": 1}
                ),
                "height": (
                    "INT",
                    {"default": 1440, "min": 1}
                ),
            },
        }
    RETURN_NAMES = ("RESIZE",)
    RETURN_TYPES = ("UPSCALER_TRT_RESIZE",)
    CATEGORY = "TensorRT/Upscaler"
    DESCRIPTION = "Specify a custom width and height for resizing."
    FUNCTION = "resize"

    def resize(self, width, height):
        return ({"width": width, "height": height},)

class ResizePresetNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "resolution": (
                    ["HD", "FullHD", "2K", "4K"],
                    {"default": "FullHD", "tooltip": "HD: 1280x720, FullHD: 1920x1080, 2K: 2560x1440, 4K: 3840x2160"}
                ),
            },
        }
    RETURN_NAMES = ("RESIZE",)
    RETURN_TYPES = ("UPSCALER_TRT_RESIZE",)
    CATEGORY = "TensorRT/Upscaler"
    DESCRIPTION = "Specify a custom width and height for resizing."
    FUNCTION = "resize"

    def resize(self, resolution):
        match resolution:
            case "HD":
                width = 1280
                height = 720
            case "FullHD":
                width = 1920
                height = 1080
            case "2K":
                width = 2560
                height = 1440
            case "4K":
                width = 3840
                height = 2160
            case _:
                raise ValueError("Invalid resolution specified")
        return ({"width": width, "height": height},)


class LoaderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    ["4x-AnimeSharp", "4x-UltraSharp", "4x-WTP-UDS-Esrgan", "4x_NMKD-Siax_200k", "4x_RealisticRescaler_100000_G", "4x_foolhardy_Remacri", "RealESRGAN_x4", "4xNomos2_otf_esrgan"],
                    {"default": "4x-UltraSharp", "tooltip": "These models have been tested with TensorRT"}
                ),
                "precision": (
                    ["fp16", "fp32"],
                    {"default": "fp16", "tooltip": "Precision to build the TensorRT engines"}
                ),
            },
            "optional": {
                "options": (
                    "UPSCALER_TRT_ENGINE_OPTIONS",
                    {"tooltip": "Options for building the TensorRT engine"}
                ),
            }
        }
    RETURN_NAMES = ("ENGINE",)
    RETURN_TYPES = ("UPSCALER_TRT_ENGINE_PKG",)
    CATEGORY = "TensorRT/Upscaler"
    DESCRIPTION = "Load TensorRT model (the model will be built automatically if not found)"
    FUNCTION = "load_upscaler_tensorrt_model"

    def load_upscaler_tensorrt_model(self, model, precision, options=None):
        if options is None:
            options = EngineBuildOptions()

        tensorrt_models_dir = os.path.join(folder_paths.models_dir, "tensorrt", "upscaler")
        onnx_models_dir = os.path.join(folder_paths.models_dir, "onnx")

        os.makedirs(tensorrt_models_dir, exist_ok=True)
        os.makedirs(onnx_models_dir, exist_ok=True)

        onnx_model_path = os.path.join(onnx_models_dir, f"{model}.onnx")

        # Engine config, should this power be given to people to decide?
        engine_channel = 3
        engine_min_batch, engine_opt_batch, engine_max_batch = options.batch_min, options.batch_opt, options.batch_max
        engine_min_h, engine_opt_h, engine_max_h = options.height_min, options.height_opt, options.height_max
        engine_min_w, engine_opt_w, engine_max_w = options.width_min, options.width_opt, options.width_max
        tensorrt_model_path = os.path.join(tensorrt_models_dir, f"{model}_{precision}_{engine_min_batch}x{engine_channel}x{engine_min_h}x{engine_min_w}_{engine_opt_batch}x{engine_channel}x{engine_opt_h}x{engine_opt_w}_{engine_max_batch}x{engine_channel}x{engine_max_h}x{engine_max_w}_{tensorrt.__version__}.trt")

        # Download onnx & build tensorrt engine
        if not os.path.exists(tensorrt_model_path):
            if not os.path.exists(onnx_model_path):
                onnx_model_download_url = f"https://huggingface.co/hekmon/ComfyUI-Upscaler-Onnx/resolve/main/{model}.onnx"
                logger.info(f"Downloading {onnx_model_download_url}")
                download_file(url=onnx_model_download_url, save_path=onnx_model_path)
            else:
                logger.info(f"Onnx model found at: {onnx_model_path}")

            # Build tensorrt engine
            logger.info(f"Building TensorRT engine for {onnx_model_path}: {tensorrt_model_path}")
            mm.soft_empty_cache()
            s = time.time()
            engine = Engine(tensorrt_model_path)
            result = engine.build(
                onnx_path=onnx_model_path,
                fp16= True if precision == "fp16" else False, # mixed precision not working TODO: investigate
                input_profile=[
                    {"input": [(engine_min_batch,engine_channel,engine_min_h,engine_min_w), (engine_opt_batch,engine_channel,engine_opt_h,engine_min_w), (engine_max_batch,engine_channel,engine_max_h,engine_max_w)]}, # any sizes from 256x256 to 1280x1280
                ],
            )
            if result != 0:
                raise Exception("Failed to build the engine. Please check the console.")
            e = time.time()
            logger.info(f"Time taken to build: {(e-s)} seconds")

        # Load tensorrt model
        logger.info(f"Loading TensorRT engine: {tensorrt_model_path}")
        mm.soft_empty_cache()
        engine = Engine(tensorrt_model_path)
        engine.load()

        return ({"engine": engine, "options": options},)

class EngineBuildOptions:
    width_min: int = 256
    width_opt: int = 512
    width_max: int = 1280
    height_min: int = 256
    height_opt: int = 512
    height_max: int = 1280
    batch_min: int = 1
    batch_opt: int = 1
    batch_max: int = 1


class EngineBuildOptionsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width_min" : (
                    "INT",
                    {"default": EngineBuildOptions.width_min, "min": 1, "max": 4096, "tooltip": "Minimal width the TensorRT engine will accept as input"}
                ),
                "width_opt" : (
                    "INT",
                    {"default": EngineBuildOptions.width_opt, "min": 1, "max": 4096, "tooltip": "Optimal width the TensorRT engine will accept as input"}
                ),
                "width_max" : (
                    "INT",
                    {"default": EngineBuildOptions.width_max, "min": 1, "max": 4096, "tooltip": "Maximum width the TensorRT engine will accept as input"}
                ),
                "height_min" : (
                    "INT",
                    {"default": EngineBuildOptions.height_min, "min": 1, "max": 4096, "tooltip": "Minimal height the TensorRT engine will accept as input"}
                ),
                "height_opt" : (
                    "INT",
                    {"default": EngineBuildOptions.height_opt, "min": 1, "max": 4096, "tooltip": "Optimal height the TensorRT engine will accept as input"}
                ),
                "height_max" : (
                    "INT",
                    {"default": EngineBuildOptions.height_max, "min": 1, "max": 4096, "tooltip": "Maximum height the TensorRT engine will accept as input"}
                ),
                "batch_min" : (
                    "INT",
                    {"default": EngineBuildOptions.batch_min, "min": 1, "tooltip": "Minimal batch size the TensorRT engine will accept as input"}
                ),
                "batch_opt" : (
                    "INT",
                    {"default": EngineBuildOptions.batch_opt, "min": 1, "tooltip": "Optimal batch size the TensorRT engine will accept as input"}
                ),
                "batch_max" : (
                    "INT",
                    {"default": EngineBuildOptions.batch_max, "min": 1, "tooltip": "Maximum batch size the TensorRT engine will accept as input"}
                ),
            },
        }

    RETURN_TYPES = ("UPSCALER_TRT_ENGINE_OPTIONS",)
    RETURN_NAMES = ("OPTIONS",)
    CATEGORY = "TensorRT/Upscaler"
    FUNCTION = "package"

    @classmethod
    def VALIDATE_INPUTS(cls, width_min, width_opt, width_max, height_min, height_opt, height_max, batch_min, batch_opt, batch_max):
        if width_min is not None and width_min < 1:
            return "width_min can not be lower than 1"
        if width_opt is not None and width_opt < 1:
            return "width_opt can not be lower than 1"
        if width_max is not None and width_max < 1:
            return "width_max can not be lower than 1"
        if width_min is not None and width_opt is not None and width_min > width_opt:
            return "width_min should not be greater than width_opt"
        if width_opt is not None and width_opt is not None and width_opt > width_max:
            return "width_opt should not be greater than width_max"

        if height_min is not None and height_min < 1:
            return "height_min can not be lower than 1"
        if height_opt is not None and height_opt < 1:
            return "height_opt can not be lower than 1"
        if height_max is not None and height_max < 1:
            return "height_max can not be lower than 1"
        if height_min is not None and height_opt is not None and height_min > height_opt:
            return "height_min should not be greater than height_opt"
        if height_opt is not None and height_max is not None and height_opt > height_max:
            return "height_opt should not be greater than height_max"

        if batch_min is not None and batch_min < 1:
            return "batch_min can not be lower than 1"
        if batch_opt is not None and batch_opt < 1:
            return "batch_opt can not be lower than 1"
        if batch_max is not None and batch_max < 1:
            return "batch_max can not be lower than 1"
        if batch_min is not None and batch_opt is not None and batch_min > batch_opt:
            return "batch_min should not be greater than batch_opt"
        if batch_opt is not None and batch_max is not None and batch_opt > batch_max:
            return "batch_opt should not be greater than batch_max"

        return True

    def package(self, width_min, width_opt, width_max, height_min, height_opt, height_max, batch_min, batch_opt, batch_max):
        options = EngineBuildOptions()
        options.width_min = width_min
        options.width_opt = width_opt
        options.width_max = width_max
        options.height_min = height_min
        options.height_opt = height_opt
        options.height_max = height_max
        options.batch_min = batch_min
        options.batch_opt = batch_opt
        options.batch_max = batch_max
        return (options,)


NODE_CLASS_MAPPINGS = {
    "UpscalerTensorRT": UpscalerNode,
    "UpscalerTensorRTResize": ResizeNode,
    "UpscalerTensorRTResizePreset": ResizePresetNode,
    "UpscalerTensorRTLoader": LoaderNode,
    "UpscalerTensorRTEngineBuildOptions": EngineBuildOptionsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscalerTensorRT": "Upscaler⚡",
    "UpscalerTensorRTResize": "Resize Custom",
    "UpscalerTensorRTResizePreset": "Resize Preset",
    "UpscalerTensorRTLoader": "Loader",
    "UpscalerTensorRTEngineBuildOptions": "Engine Build Options",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
