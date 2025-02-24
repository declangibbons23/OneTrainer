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

    def _enumerate_input_modules(self, config, is_validation: bool = False):
        """Create input enumeration modules"""
        return self._create_pipeline_definition(
            config=config,
            cache_dir=config.cache_dir,
            concepts=config.concepts,
            is_validation=is_validation
        )

    def _load_input_modules(self, config, train_dtype, add_embeddings_to_prompt):
        """Create input loading modules"""
        modules = []

        # Add text processing modules
        if add_embeddings_to_prompt:
            modules.extend([
                SelectRandomText(
                    texts_in_name='texts',
                    text_out_name='text'
                ),
                CapitalizeTags(
                    text_in_name='text',
                    text_out_name='text'
                ),
                ShuffleTags(
                    text_in_name='text',
                    text_out_name='text'
                ),
                DropTags(
                    text_in_name='text',
                    text_out_name='text'
                ),
            ])

        # Add image processing modules
        modules.extend([
            LoadImage(
                path_in_name='image_path',
                image_out_name='image',
                range_min=0.0,
                range_max=1.0,
                supported_extensions=['.jpg', '.jpeg', '.png', '.webp']
            ),
            GetFilename(
                path_in_name='image_path',
                filename_out_name='filename'
            ),
        ])

        return modules

    def _mask_augmentation_modules(self, config):
        """Create mask augmentation modules"""
        if config.masked_training:
            return [
                RandomMaskRotateCrop(
                    mask_in_name='mask',
                    mask_out_name='mask'
                ),
                RandomCircularMaskShrink(
                    mask_in_name='mask',
                    mask_out_name='mask'
                ),
                RandomLatentMaskRemove(
                    mask_in_name='mask',
                    mask_out_name='mask'
                ),
            ]
        return []

    def _aspect_bucketing_in(self, config, bucket_side_length):
        """Create aspect bucketing input modules"""
        if config.aspect_ratio_bucketing:
            return [
                CalcAspect(
                    image_in_name='image',
                    aspect_out_name='aspect'
                ),
                AspectBucketing(
                    aspect_in_name='aspect',
                    bucket_out_name='bucket',
                    bucket_side_length=bucket_side_length
                ),
                InlineAspectBatchSorting(
                    bucket_in_name='bucket',
                    aspect_in_name='aspect'
                ),
            ]
        return []

    def _crop_modules(self, config):
        """Create crop modules"""
        return [ScaleCropImage(
            image_in_name='image',
            image_out_name='image',
            original_size_out_name='original_resolution',
            crop_size_out_name='crop_resolution',
            offset_out_name='crop_offset'
        )]

    def _augmentation_modules(self, config):
        """Create augmentation modules"""
        return [
            RandomRotate(
                image_in_name='image',
                image_out_name='image'
            ),
            RandomFlip(
                image_in_name='image',
                image_out_name='image'
            ),
            RandomBrightness(
                image_in_name='image',
                image_out_name='image'
            ),
            RandomContrast(
                image_in_name='image',
                image_out_name='image'
            ),
            RandomHue(
                image_in_name='image',
                image_out_name='image'
            ),
            RandomSaturation(
                image_in_name='image',
                image_out_name='image'
            ),
        ]

    def _inpainting_modules(self, config):
        """Create inpainting modules"""
        if config.model_type.has_conditioning_image_input():
            return [GenerateImageLike(
                image_in_name='image',
                image_out_name='conditioning_image'
            )]
        return []

    def _output_modules_from_out_names(self, output_names, sort_names, config, before_cache_image_fun, use_conditioning_image, vae, autocast_context, train_dtype):
        """Create output modules from output names"""
        modules = []

        # Add data mapping module
        modules.append(MapData(output_names))

        # Add selection module
        if use_conditioning_image:
            modules.append(SelectInput(
                image_in_name='image',
                conditioning_image_in_name='conditioning_image',
                image_out_name='image'
            ))

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
            concepts: List[Any],
            is_validation: bool = False,
    ) -> List[OutputPipelineModule]:
        """Create pipeline definition with distributed support"""
        definition = []

        # Add collect paths module
        definition.append(CollectPaths(
            concept_in_name='concept',
            path_in_name='concept.image',
            include_subdirectories_in_name='concept.include_subdirectories',
            enabled_in_name='concept.enabled',
            path_out_name='image_path',
            concept_out_name='concept',
            extensions=['.jpg', '.jpeg', '.png', '.webp'],
            include_postfix=[],
            exclude_postfix=[],
        ))

        # Add disk cache if enabled
        if config.latent_caching:
            definition.append(self._create_disk_cache(config, cache_dir))

        # Add text loading module
        definition.append(LoadMultipleTexts(
            path_in_name='concept.text',
            texts_out_name='prompt'
        ))

        # Add image loading and processing modules
        definition.append(LoadImage(
            path_in_name='image_path',
            image_out_name='image',
            range_min=0.0,
            range_max=1.0,
            supported_extensions=['.jpg', '.jpeg', '.png', '.webp']
        ))
        definition.append(GetFilename(
            path_in_name='image_path',
            filename_out_name='filename'
        ))

        # Add aspect ratio modules if enabled
        if config.aspect_ratio_bucketing:
            definition.append(CalcAspect(
                image_in_name='image',
                aspect_out_name='aspect'
            ))
            if is_validation:
                definition.append(SingleAspectCalculation(
                    aspect_in_name='aspect',
                    aspect_out_name='aspect'
                ))
            else:
                definition.append(AspectBucketing(
                    aspect_in_name='aspect',
                    bucket_out_name='bucket',
                    bucket_side_length=64
                ))
                definition.append(AspectBatchSorting(
                    bucket_in_name='bucket',
                    aspect_in_name='aspect'
                ))

        # Add augmentation modules if not validation
        if not is_validation:
            if config.masked_training:
                definition.append(RandomMaskRotateCrop(
                    path_in_name='concept.mask',
                    mask_out_name='mask'
                ))
            else:
                definition.append(RandomRotate(
                    image_in_name='image',
                    image_out_name='image'
                ))
                definition.append(RandomFlip(
                    image_in_name='image',
                    image_out_name='image'
                ))

            definition.append(RandomBrightness(
                image_in_name='image',
                image_out_name='image'
            ))
            definition.append(RandomContrast(
                image_in_name='image',
                image_out_name='image'
            ))
            definition.append(RandomHue(
                image_in_name='image',
                image_out_name='image'
            ))
            definition.append(RandomSaturation(
                image_in_name='image',
                image_out_name='image'
            ))

        # Add conditioning image modules if needed
        if config.model_type.has_conditioning_image_input():
            definition.append(GenerateMaskedConditioningImage(
                path_in_name='concept.conditioning_image',
                image_out_name='conditioning_image'
            ))

        # Add final processing modules
        definition.append(ScaleCropImage(
            image_in_name='image',
            image_out_name='image',
            original_size_out_name='original_resolution',
            crop_size_out_name='crop_resolution',
            offset_out_name='crop_offset'
        ))

        return definition
