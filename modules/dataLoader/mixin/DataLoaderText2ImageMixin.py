from typing import List, Dict, Any

from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.pipelineModules.AspectBatchSorting import AspectBatchSorting
from mgds.pipelineModules.AspectBucketing import AspectBucketing
from mgds.pipelineModules.CalcAspect import CalcAspect
from mgds.pipelineModules.CapitalizeTags import CapitalizeTags
from mgds.pipelineModules.CollectPaths import CollectPaths
from mgds.pipelineModules.DiskCache import DiskCache
from mgds.pipelineModules.DistributedDiskCache import DistributedDiskCache
from mgds.pipelineModules.DropTags import DropTags
from mgds.pipelineModules.GenerateImageLike import GenerateImageLike
from mgds.pipelineModules.GenerateMaskedConditioningImage import GenerateMaskedConditioningImage
from mgds.pipelineModules.GetFilename import GetFilename
from mgds.pipelineModules.ImageToVideo import ImageToVideo
from mgds.pipelineModules.InlineAspectBatchSorting import InlineAspectBatchSorting
from mgds.pipelineModules.LoadImage import LoadImage
from mgds.pipelineModules.LoadMultipleTexts import LoadMultipleTexts
from mgds.pipelineModules.LoadVideo import LoadVideo
from mgds.pipelineModules.MapData import MapData
from mgds.pipelineModules.ModifyPath import ModifyPath
from mgds.pipelineModules.RandomBrightness import RandomBrightness
from mgds.pipelineModules.RandomCircularMaskShrink import RandomCircularMaskShrink
from mgds.pipelineModules.RandomContrast import RandomContrast
from mgds.pipelineModules.RandomFlip import RandomFlip
from mgds.pipelineModules.RandomHue import RandomHue
from mgds.pipelineModules.RandomLatentMaskRemove import RandomLatentMaskRemove
from mgds.pipelineModules.RandomMaskRotateCrop import RandomMaskRotateCrop
from mgds.pipelineModules.RandomRotate import RandomRotate
from mgds.pipelineModules.RandomSaturation import RandomSaturation
from mgds.pipelineModules.ScaleCropImage import ScaleCropImage
from mgds.pipelineModules.SelectInput import SelectInput
from mgds.pipelineModules.SelectRandomText import SelectRandomText
from mgds.pipelineModules.ShuffleTags import ShuffleTags
from mgds.pipelineModules.SingleAspectCalculation import SingleAspectCalculation

import torch


class DataLoaderText2ImageMixin:
    """Mixin class for text-to-image data loaders with distributed support"""

    def _enumerate_input_modules(self, config):
        """Create input enumeration modules"""
        return self._create_pipeline_definition(
            config=config,
            cache_dir=config.cache_dir,
            image_paths=[concept.image for concept in config.concepts],
            text_paths=[concept.text for concept in config.concepts],
            mask_paths=[concept.mask for concept in config.concepts] if config.masked_training else None,
            conditioning_image_paths=[concept.conditioning_image for concept in config.concepts] if config.model_type.has_conditioning_image_input() else None,
        )

    def _load_input_modules(self, config, train_dtype, add_embeddings_to_prompt):
        """Create input loading modules"""
        modules = []

        # Add text processing modules
        if add_embeddings_to_prompt:
            modules.extend([
                SelectRandomText(),
                CapitalizeTags(),
                ShuffleTags(),
                DropTags(),
            ])

        # Add image processing modules
        modules.extend([
            LoadImage(),
            GetFilename(),
        ])

        return modules

    def _mask_augmentation_modules(self, config):
        """Create mask augmentation modules"""
        if config.masked_training:
            return [
                RandomMaskRotateCrop(),
                RandomCircularMaskShrink(),
                RandomLatentMaskRemove(),
            ]
        return []

    def _aspect_bucketing_in(self, config, bucket_side_length):
        """Create aspect bucketing input modules"""
        if config.aspect_ratio_bucketing:
            return [
                CalcAspect(),
                AspectBucketing(bucket_side_length=bucket_side_length),
                InlineAspectBatchSorting(),
            ]
        return []

    def _crop_modules(self, config):
        """Create crop modules"""
        return [ScaleCropImage()]

    def _augmentation_modules(self, config):
        """Create augmentation modules"""
        return [
            RandomRotate(),
            RandomFlip(),
            RandomBrightness(),
            RandomContrast(),
            RandomHue(),
            RandomSaturation(),
        ]

    def _inpainting_modules(self, config):
        """Create inpainting modules"""
        if config.model_type.has_conditioning_image_input():
            return [GenerateImageLike()]
        return []

    def _output_modules_from_out_names(self, output_names, sort_names, config, before_cache_image_fun, use_conditioning_image, vae, autocast_context, train_dtype):
        """Create output modules from output names"""
        modules = []

        # Add data mapping module
        modules.append(MapData(output_names))

        # Add selection module
        if use_conditioning_image:
            modules.append(SelectInput())

        return modules

    def _create_disk_cache(self, config, cache_dir: str) -> DiskCache | DistributedDiskCache:
        """Create appropriate disk cache based on FSDP setting"""
        if config.enable_fsdp:
            # Get distributed info
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            
            return DistributedDiskCache(
                cache_dir=cache_dir,
                world_size=world_size,
                rank=rank
            )
        else:
            return DiskCache(cache_dir=cache_dir)

    def _create_pipeline_definition(
            self,
            config,
            cache_dir: str,
            image_paths: List[str],
            text_paths: List[str],
            mask_paths: List[str] = None,
            conditioning_image_paths: List[str] = None,
            is_validation: bool = False,
    ) -> List[OutputPipelineModule]:
        """Create pipeline definition with distributed support"""
        definition = []

        # Add collect paths module
        definition.append(CollectPaths(image_paths))

        # Add disk cache if enabled
        if config.latent_caching:
            definition.append(self._create_disk_cache(config, cache_dir))

        # Add text loading module
        if text_paths:
            definition.append(LoadMultipleTexts(text_paths))

        # Add image loading and processing modules
        definition.append(LoadImage())
        definition.append(GetFilename())

        # Add aspect ratio modules if enabled
        if config.aspect_ratio_bucketing:
            definition.append(CalcAspect())
            if is_validation:
                definition.append(SingleAspectCalculation())
            else:
                definition.append(AspectBucketing())
                definition.append(AspectBatchSorting())

        # Add augmentation modules if not validation
        if not is_validation:
            if config.masked_training and mask_paths:
                definition.append(RandomMaskRotateCrop(mask_paths))
            else:
                definition.append(RandomRotate())
                definition.append(RandomFlip())

            definition.append(RandomBrightness())
            definition.append(RandomContrast())
            definition.append(RandomHue())
            definition.append(RandomSaturation())

        # Add conditioning image modules if needed
        if conditioning_image_paths:
            definition.append(GenerateMaskedConditioningImage(conditioning_image_paths))

        # Add final processing modules
        definition.append(ScaleCropImage())

        return definition
