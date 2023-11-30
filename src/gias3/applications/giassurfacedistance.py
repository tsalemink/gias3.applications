#!/usr/bin/env python
"""
FILE: giassurfacedistance.py
LAST MODIFIED: 19/03/18
DESCRIPTION:
Script for calculating the distances between 2 surfaces.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""
import logging

import argparse
from argparse import RawTextHelpFormatter
from os import path

import numpy as np
import sys
from scipy.spatial import cKDTree
from scipy.spatial.distance import jaccard, dice, directed_hausdorff

from gias3.mesh import vtktools

log = logging.getLogger(__name__)

try:
    from gias3.visualisation import fieldvi

    can_visual = True
except (ImportError, NotImplementedError):
    log.info('no visualisation available')
    can_visual = False

_descStr = """Script for calculating the distances between 2 surfaces.
Author: Ju Zhang
Last Modified: 2017-08-02

This script takes 2 surfaces (groundtruth and test) and calculates the Jaccard
and dice indices, and the 2-way surface to surface distances. The 2-way
distance means for each vertex on the groundtruth find the closest vertex on
the test and then vice versa. Thus a total of m+n distances are calculated,
where m and n are the number of vertices on the groundtruth and test surfaces
respectively. The Hausdorff distance is also calculated.

The max, mean, and rms of these distances and the Jaccard index are printed to
terminal or file.

Things to note
--------------
- Surface vertex density definitely matters since only vertex-to-vertex
  distances are calculated. Sparser the points, less accurate the results
  represent the true surface to surface distance.

- Units dimension is expected to be in mm. If not, units for the groundtruth
  and test surface coordinates can be defined by the --groundtruth-unit and
  --test-unit options.

- The format of the surface files can be stl, ply, obj, vrml, or vtp.

- The results are printed to the terminal (and to file if the -o option is
  specified), e.g.:

    dhausdorff: 0.719000284750812
    dice: 0.9776663756442392
    dmax: 0.789712039703775
    dmean: 0.13066467628268588
    drms: 0.17495242536528388
    jaccard: 0.9563085399449036

"""

def dim_unit_scaling(in_unit, out_unit):
    """
    Calculate the scaling factor to convert from the input unit (in_unit) to
    the output unit (out_unit). in_unit and out_unit must be a string and one
    of ['nm', 'um', 'mm', 'cm', 'm', 'km']. 

    inputs
    ======
    in_unit : str
        Input unit
    out_unit :str
        Output unit

    returns
    =======
    scaling_factor : float
    """

    unit_vals = {
        'nm': 1e-9,
        'um': 1e-6,
        'mm': 1e-3,
        'cm': 1e-2,
        'm': 1.0,
        'km': 1e3,
    }

    if in_unit not in unit_vals:
        raise ValueError(
            'Invalid input unit {}. Must be one of {}'.format(
                in_unit, list(unit_vals.keys())
            )
        )
    if out_unit not in unit_vals:
        raise ValueError(
            'Invalid input unit {}. Must be one of {}'.format(
                in_unit, list(unit_vals.keys())
            )
        )

    return unit_vals[in_unit] / unit_vals[out_unit]


def _rms(x):
    return np.sqrt((x * x).mean())


def loadMesh(filename):
    r = vtktools.Reader()
    r.read(filename)
    return r.getSimplemesh()


def calcOverlap(s1, s2, orig, shape, spacing):
    I1 = vtktools.triSurface2BinaryMask(
        s1.v, s1.f, shape, orig, spacing
    )[0].astype(int)
    I2 = vtktools.triSurface2BinaryMask(
        s2.v, s2.f, shape, orig, spacing
    )[0].astype(int)
    j = 1.0 - jaccard(I1.ravel(), I2.ravel())
    d = 1.0 - dice(I1.ravel(), I2.ravel())
    return j, d, I1, I2


def calcDistance(s1, s2):
    tree1 = cKDTree(s1.v)
    tree2 = cKDTree(s2.v)
    d21, d21i = tree1.query(s2.v, k=1)
    d12, d12i = tree2.query(s1.v, k=1)

    dhaus = directed_hausdorff(s1.v, s2.v)[0]
    # dmax = max([max(d21), max(d12)])
    # drmsmean = np.mean([_rms(d21), _rms(d12)])
    # dmeanmean = np.mean([d21.mean(), d12.mean()])

    dmax = np.hstack([d21, d12]).max()
    drms = _rms(np.hstack([d21, d12]))
    dmean = np.hstack([d21, d12]).mean()

    return dmax, drms, dmean, dhaus, (d12, d21)


def calcSegmentationErrors(mesh_file_test, mesh_file_gt, jac_img_spacing, gt_scaling, test_scaling):
    # load ground truth segmentation (tri-mesh)
    surfGT = loadMesh(mesh_file_gt)
    surfGT.v *= gt_scaling

    # load test segmentation (tri-mesh)
    surfTest = loadMesh(mesh_file_test)
    surfTest.v *= test_scaling

    # work out volume size
    volMin = np.min([surfGT.v.min(0), surfTest.v.min(0)], axis=0)
    volMax = np.max([surfGT.v.max(0), surfTest.v.max(0)], axis=0)
    imgOrig = volMin - 10.0
    imgShape = np.ceil(((volMax + 10.0) - imgOrig) / jac_img_spacing).astype(int)

    # calc jaccard coeff
    jaccard_, dice_, imgGT, imgTest = calcOverlap(surfGT, surfTest, imgOrig, imgShape, jac_img_spacing)

    # calc surface to surface distance
    dmax, drms, dmean, dhaus, (d12, d21) = calcDistance(surfGT, surfTest)

    results = {'jaccard': jaccard_,
               'dice': dice_,
               'dmax': dmax,
               'drms': drms,
               'dmean': dmean,
               'dhausdorff': dhaus,
               }

    return results, surfTest, surfGT, imgTest, imgGT


def visualise(V, surf_test, surf_gt, img_test, img_gt):
    V.addTri('test surface', surf_test, render_args={'color': (0.4, 0.4, 0.4)})
    V.updateTriSurface('test surface')
    V.addTri('ground truth surface', surf_gt, render_args={'color': (0.84705882, 0.8, 0.49803922)})
    V.updateTriSurface('ground truth surface')
    V.addImageVolume(img_gt.astype(float), 'groundtruth')
    V.addImageVolume(img_test.astype(float), 'test')


def writeResults(filepath, testname, gtname, res):
    text = 'groundtruth: {}, test: {}, rmsd: {:9.6f}, meand: {:9.6f}, maxd: {:9.6f}, hausdorff: {:9.6f}, jaccard{:9.6f}, dice{:9.6f}\n'
    with open(filepath, 'a') as f:
        f.write(
            text.format(
                gtname,
                testname,
                res['drms'],
                res['dmean'],
                res['dmax'],
                res['dhausdorff'],
                res['jaccard'],
                res['dice'],
            )
        )


def main():
    # =============================================================================#
    imgSpacing = np.array([0.5, ] * 3, dtype=float)
    unitChoices = ('nm', 'um', 'mm', 'cm', 'm', 'km')
    defaultUnit = 'mm'
    # =============================================================================#
    parser = argparse.ArgumentParser(
        description=_descStr,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument('gTruthPath',
                        help='ground truth surface path')
    parser.add_argument('testPath',
                        help='test surface path')
    parser.add_argument('-o', '--outpath',
                        help='results output path. Results are append to text file')
    parser.add_argument('-d', '--display',
                        action='store_true',
                        help='visualise results')
    parser.add_argument('--groundtruth-unit',
                        action='store',
                        default='mm',
                        choices=unitChoices,
                        help='unit of ground truth coordinates')
    parser.add_argument('--test-unit',
                        action='store',
                        default='mm',
                        choices=unitChoices,
                        help='unit of test coordinates')

    args = parser.parse_args()
    gtUnitScaling = dim_unit_scaling(args.groundtruth_unit, defaultUnit)
    testUnitScaling = dim_unit_scaling(args.test_unit, defaultUnit)
    results, surfTest, surfGT, imgTest, imgGT = calcSegmentationErrors(
        args.testPath,
        args.gTruthPath,
        imgSpacing,
        gtUnitScaling,
        testUnitScaling
    )
    for k, v in sorted(results.items()):
        log.info('{}: {}'.format(k, v))

    if args.outpath is not None:
        testName = path.splitext(path.split(args.testPath)[1])[0]
        gtName = path.splitext(path.split(args.gTruthPath)[1])[0]
        writeResults(args.outpath, testName, gtName, results)

    if args.display and can_visual:
        V = fieldvi.FieldVi()
        V.start()
        V.scene.background = (0, 0, 0)
        visualise(V, surfTest, surfGT, imgTest, imgGT)


if __name__ == '__main__':
    main()
