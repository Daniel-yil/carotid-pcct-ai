# Automated Carotid Artery Analysis Pipeline

An end-to-end medical imaging pipeline for automated carotid artery
segmentation and quantitative vascular analysis from CT imaging.

The project combines nnU-Net-based segmentation with automated
post-processing and a clinician-facing analysis interface.

## Project Overview

The pipeline was developed for automated analysis of carotid arteries
from PCCT/CTA imaging.

The workflow includes:

1. CT image preprocessing
2. Multi-label carotid artery segmentation
3. Vascular centerline extraction
4. Vessel morphology analysis
5. Stenosis quantification
6. Calcification analysis
7. Perivascular fat attenuation (FAI) analysis
8. Visualization through an analysis interface

## Pipeline

CT Image
↓
Preprocessing
↓
nnU-Net Segmentation
↓
4-Label Carotid Artery Masks
↓
Post-processing
↓
Centerline Extraction
↓
Quantitative Analysis
↓
Visualization / Analysis Interface

## Segmentation

The segmentation pipeline is based on nnU-Net.

Major preprocessing steps include:

- Image and label validation
- Voxel spacing standardization
- Label remapping
- nnU-Net preprocessing
- Model training
- Automated inference

The model performs 4-label carotid artery segmentation.

## Post-processing

The post-processing pipeline uses:

- SimpleITK
- NumPy
- 3D Slicer

The pipeline automatically extracts vascular centerlines and calculates
quantitative imaging measurements.

## Quantitative Outputs

The system generates:

- Vessel diameter
- Stenosis percentage
- Calcification measurements
- Perivascular fat attenuation (FAI)
- Vascular morphology statistics
- Segmentation masks
- Calcification maps
- FAI maps
