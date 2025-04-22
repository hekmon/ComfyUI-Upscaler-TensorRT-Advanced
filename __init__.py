import os
import folder_paths
import numpy as np
import torch
from comfy.utils import ProgressBar
from .trt_utilities import Engine
from .utilities import download_file, ColoredLogger, get_final_resolutions
import comfy.model_management as mm
import time
import tensorrt

logger = ColoredLogger("ComfyUI-Upscaler-Tensorrt")


class UpscalerTensorrt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {"tooltip": "Images to be upscaled. Resolution must be between 256 and 1280 px"}
                ),
                "upscaler_trt_model": (
                    "UPSCALER_TRT_MODEL",
                    {"tooltip": "Tensorrt model built and loaded"}
                ),
                "resize_to": (
                    ["none", "HD", "FHD", "2k", "4k"],
                    {"tooltip": "Resize the upscaled image to fixed resolutions, optional"}
                ),
            }
        }
    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscaler_tensorrt"
    CATEGORY = "TensorRT/Upscaler"
    DESCRIPTION = "Upscale images with TensorRT"

    def upscaler_tensorrt(self, images, upscaler_trt_model, resize_to):
        images_bchw = images.permute(0, 3, 1, 2)
        B, C, H, W = images_bchw.shape
        final_width, final_height = get_final_resolutions(W, H, resize_to)
        logger.info(f"Upscaling {B} images from H:{H}, W:{W} to H:{H*4}, W:{W*4} | Final resolution: H:{final_height}, W:{final_width} | resize_to: {resize_to}")

        shape_dict = {
            "input": {"shape": (1, 3, H, W)},
            "output": {"shape": (1, 3, H*4, W*4)},
        }
        # setup engine
        upscaler_trt_model.activate()
        upscaler_trt_model.allocate_buffers(shape_dict=shape_dict)

        cudaStream = torch.cuda.current_stream().cuda_stream
        pbar = ProgressBar(B)
        images_list = list(torch.split(images_bchw, split_size_or_sections=1))

        upscaled_frames = torch.empty((B, C, final_height, final_width), dtype=torch.float32, device=mm.intermediate_device()) # offloaded to cpu
        must_resize = W*4 != final_width or H*4 != final_height

        for i, img in enumerate(images_list):
            result = upscaler_trt_model.infer({"input": img}, cudaStream)
            result = result["output"]

            if must_resize:
                result = torch.nn.functional.interpolate(
                    result,
                    size=(final_height, final_width),
                    mode='bicubic',
                    antialias=True
                )
            upscaled_frames[i] = result.to(mm.intermediate_device())
            pbar.update(1)

        output = upscaled_frames.permute(0, 2, 3, 1)
        upscaler_trt_model.reset() # frees engine vram
        mm.soft_empty_cache()

        logger.info(f"Output shape: {output.shape}")
        return (output,)


class LoadUpscalerTensorrtModel:
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
                "build_options": (
                    "ENGINE_OPTIONS",
                    {"tooltip": "Options for building the TensorRT engine"}
                ),
            }
        }
    RETURN_NAMES = ("upscaler_trt_model",)
    RETURN_TYPES = ("UPSCALER_TRT_MODEL",)
    CATEGORY = "TensorRT/Upscaler"
    DESCRIPTION = "Load TensorRT model (the model will be built automatically if not found)"
    FUNCTION = "load_upscaler_tensorrt_model"

    def load_upscaler_tensorrt_model(self, model, precision, build_options=None):
        if build_options is None:
            build_options = EngineBuildOptions()

        tensorrt_models_dir = os.path.join(folder_paths.models_dir, "tensorrt", "upscaler")
        onnx_models_dir = os.path.join(folder_paths.models_dir, "onnx")

        os.makedirs(tensorrt_models_dir, exist_ok=True)
        os.makedirs(onnx_models_dir, exist_ok=True)

        onnx_model_path = os.path.join(onnx_models_dir, f"{model}.onnx")

        # Engine config, should this power be given to people to decide?
        engine_channel = 3
        engine_min_batch, engine_opt_batch, engine_max_batch = build_options.batch_min, build_options.batch_opt, build_options.batch_max
        engine_min_h, engine_opt_h, engine_max_h = build_options.height_min, build_options.height_opt, build_options.height_max
        engine_min_w, engine_opt_w, engine_max_w = build_options.width_min, build_options.width_opt, build_options.width_max
        tensorrt_model_path = os.path.join(tensorrt_models_dir, f"{model}_{precision}_{engine_min_batch}x{engine_channel}x{engine_min_h}x{engine_min_w}_{engine_opt_batch}x{engine_channel}x{engine_opt_h}x{engine_opt_w}_{engine_max_batch}x{engine_channel}x{engine_max_h}x{engine_max_w}_{tensorrt.__version__}.trt")

        # Download onnx & build tensorrt engine
        if not os.path.exists(tensorrt_model_path):
            if not os.path.exists(onnx_model_path):
                onnx_model_download_url = f"https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/{model}.onnx"
                logger.info(f"Downloading {onnx_model_download_url}")
                download_file(url=onnx_model_download_url, save_path=onnx_model_path)
            else:
                logger.info(f"Onnx model found at: {onnx_model_path}")

            # Build tensorrt engine
            logger.info(f"Building TensorRT engine for {onnx_model_path}: {tensorrt_model_path}")
            mm.soft_empty_cache()
            s = time.time()
            engine = Engine(tensorrt_model_path)
            engine.build(
                onnx_path=onnx_model_path,
                fp16= True if precision == "fp16" else False, # mixed precision not working TODO: investigate
                input_profile=[
                    {"input": [(engine_min_batch,engine_channel,engine_min_h,engine_min_w), (engine_opt_batch,engine_channel,engine_opt_h,engine_min_w), (engine_max_batch,engine_channel,engine_max_h,engine_max_w)]}, # any sizes from 256x256 to 1280x1280
                ],
            )
            e = time.time()
            logger.info(f"Time taken to build: {(e-s)} seconds")

        # Load tensorrt model
        logger.info(f"Loading TensorRT engine: {tensorrt_model_path}")
        mm.soft_empty_cache()
        engine = Engine(tensorrt_model_path)
        engine.load()

        return (engine,)


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
                    {"default": 256, "min": 1, "max": 4096, "tooltip": "Minimal width the TensorRT engine will accept as input"}
                ),
                "width_opt" : (
                    "INT",
                    {"default": 512, "min": 1, "max": 4096, "tooltip": "Optimal width the TensorRT engine will accept as input"}
                ),
                "width_max" : (
                    "INT",
                    {"default": 1280, "min": 1, "max": 4096, "tooltip": "Maximum width the TensorRT engine will accept as input"}
                ),
                "height_min" : (
                    "INT",
                    {"default": 256, "min": 1, "max": 4096, "tooltip": "Minimal height the TensorRT engine will accept as input"}
                ),
                "height_opt" : (
                    "INT",
                    {"default": 512, "min": 1, "max": 4096, "tooltip": "Optimal height the TensorRT engine will accept as input"}
                ),
                "height_max" : (
                    "INT",
                    {"default": 1280, "min": 1, "max": 4096, "tooltip": "Maximum height the TensorRT engine will accept as input"}
                ),
                "batch_min" : (
                    "INT",
                    {"default": 1, "min": 1, "tooltip": "Minimal batch size the TensorRT engine will accept as input"}
                ),
                "batch_opt" : (
                    "INT",
                    {"default": 1, "min": 1, "tooltip": "Optimal batch size the TensorRT engine will accept as input"}
                ),
                "batch_max" : (
                    "INT",
                    {"default": 1, "min": 1, "tooltip": "Maximum batch size the TensorRT engine will accept as input"}
                ),
            },
        }

    RETURN_TYPES = ("ENGINE_OPTIONS",)
    RETURN_NAMES = ("BUILD_OPTIONS",)
    CATEGORY = "TensorRT/Upscaler"
    FUNCTION = "package"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        if input_types["width_opt"] < input_types["width_min"]:
            return "width_opt must be greater than or equal to width_min"
        if input_types["width_opt"] > input_types["width_max"]:
            return "width_opt must be less than or equal to width_max"
        if input_types["height_opt"] < input_types["height_min"]:
            return "height_opt must be greater than or equal to height_min"
        if input_types["height_opt"] > input_types["height_max"]:
            return "height_opt must be less than or equal to height_max"
        if input_types["batch_opt"] < input_types["batch_min"]:
            return "batch_opt must be greater than or equal to batch_min"
        if input_types["batch_opt"] > input_types["batch_max"]:
            return "batch_opt must be less than or equal to batch_max"
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
    "UpscalerTensorrt": UpscalerTensorrt,
    "LoadUpscalerTensorrtModel": LoadUpscalerTensorrtModel,
    "EngineBuildOptions": EngineBuildOptionsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscalerTensorrt": "TensorRT Upscaler ⚡",
    "LoadUpscalerTensorrtModel": "TensorRT Upscale Model Loader",
    "EngineBuildOptions": "TensorRT Engine Builder Options",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
