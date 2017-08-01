#!/usr/bin/env python
"""
FILE: pcreg.py
LAST MODIFIED: 31/07/17
DESCRIPTION:
Script for registering one model to another using principal components.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

from os import path
import sys
import argparse
import numpy as np
from scipy.spatial import cKDTree
import copy
import logging

from gias2.registration import alignment_fitting as af
from gias2.registration import shapemodel
from gias2.mesh import vtktools
from gias2.learning import PCA

r2c = shapemodel.r2c13
FTOL = 1e-6

def register(mean_mesh, ssm, target, init_rot, fit_mode, fit_comps,
    mw=1.0, sample=None, pts_only=False, fit_scale=False, out=None,
    auto_align=True, view=False
    ):
    
    if pts_only:
        source_points = r2c(ssm.getMean())
        target_points = target
    else:
        source_points = mean_mesh.v
        target_points = target.v

    #=============================================================#
    # rigidly register mean points to target points
    if auto_align:
        init_trans = target_points.mean(0) - source_points.mean(0)
        t0 = np.hstack([init_trans, init_rot])
        reg1_T, source_points_reg1, reg1_errors = af.fitDataRigidDPEP(
                                                    source_points,
                                                    target_points,
                                                    xtol=1e-6,
                                                    sample=1000,
                                                    t0=t0,
                                                    outputErrors=1
                                                    )

        # add isotropic scaling to rigid registration
        if fit_scale:
            reg2_T, source_points_reg2, reg2_errors = af.fitDataRigidScaleDPEP(
                                                        source_points_reg1,
                                                        target_points,
                                                        xtol=1e-6,
                                                        sample=1000,
                                                        t0=np.hstack([reg1_T, 1.0]),
                                                        outputErrors=1
                                                        )
        else:
            reg2_T = reg1_T
    else:
        reg2_T = np.zeros(6)
        if fit_scale:
            reg2_T = np.hstack([reg2_T, 1.0])

    #=============================================================#
    # shape model registration
    reg_T3, source_points_reg3,\
    (reg_err, reg_rms, reg_mdist) = shapemodel.fitSSMTo3DPoints(
        target_points, ssm, fit_comps, fit_mode,
        mw=mw, init_t=reg2_T, fit_scale=fit_scale, ftol=FTOL, sample=sample,
        recon2coords=r2c, verbose=view
        )

    #=============================================================#
    # create regstered mesh
    if pts_only:
        reg = source_points_reg3
    else:
        reg = copy.deepcopy(mean_mesh)
        reg.v = source_points_reg3

    if out:
        if pts_only:
            n = np.arange(1,len(reg))
            _out = np.hstack([n[:,np.newaxis], reg])
            np.savetxt(
                args.out, _out, delimiter=',',
                fmt=['%6d', '%10.6f', '%10.6f', '%10.6f'],
                header='# shape model registered points'
                )
        else:
            writer = vtktools.Writer(v=reg.v, f=reg.f)
            writer.write(out)

    #=============================================================#
    # view
    if view:
        try:
            from gias2.visualisation import fieldvi
            has_mayavi = True
        except ImportError:
            has_mayavi = False

        if has_mayavi:
            v = fieldvi.Fieldvi()
            if pts_only:
                v.addData('target', target, renderArgs={'color':(1,0,0)})
                v.addData('mean', mean_mesh, renderArgs={'color':(0,1,0)})
                v.addData('mean morphed', reg, renderArgs={'color':(0.3,0.3,1)})
            else:
                v.addTri('target', target, renderArgs={'color':(1,0,0)})
                v.addTri('mean', mean_mesh, renderArgs={'color':(0,1,0)})
                v.addTri('mean morphed', reg, renderArgs={'color':(0.3,0.3,1)})
            
            # v.addData('source points reg 2', source_points_reg2, renderArgs={'mode':'point'})
            v.scene.background=(0,0,0)
            v.configure_traits()
        else:
            print('Visualisation error: cannot import mayavi')

    return reg, reg_rms

def reg_single(args, mean=None, ssm=None):
    print('{} to {}'.format(args.mean, args.target))

    if mean is None:
        if args.points_only:
            mean = np.loadtxt(args.mean, skiprows=1, use_cols=(1,2,3))
        else:
            mean = vtktools.loadpoly(args.mean)
    
    if args.points_only:
        target = np.loadtxt(args.target, skiprows=1, use_cols=(1,2,3))
    else:
        target = vtktools.loadpoly(args.target)

    if ssm is None:
        ssm = PCA.loadPrincipalComponents(args.ssm)
    
    init_rot = np.deg2rad((0,0,0))

    reg, rms = register(mean, ssm, target, init_rot, args.fit_mode,
        args.fit_comps, mw=args.mweight, sample=args.sample,
        pts_only=args.points_only, fit_scale=args.fit_scale, out=args.out,
        auto_align=args.auto_align, view=args.view
        )

    logging.info('{}, rms: {}'.format(path.split(args.target)[1], rms))

def reg_batch(args):
    model_paths = np.loadtxt(args.batch, dtype=str)

    if args.points_only:
        mean = np.loadtxt(args.mean, skiprows=1, use_cols=(1,2,3))
    else:
        mean = vtktools.loadpoly(args.mean)

    ssm = PCA.loadPrincipalComponents(args.ssm)
    
    out_dir = args.outdir
    for i, mp in enumerate(model_paths):
        args.target = mp
        _p, _ext = path.splitext(path.split(mp)[1])
        if args.outext is not None:
            _ext = args.outext
        args.out = path.join(out_dir, _p+'_ssmreg'+_ext)
        main_single(args, mean, ssm)

def main():
    parser = argparse.ArgumentParser(description='Non-rigid registration using a PCA shape model.')
    parser.add_argument(
        'mean',
        help='file path of the mean mesh or point cloud.'
        )
    parser.add_argument(
        'ssm',
        help='file path of the shape model (.pc or .pc.npz file).'
        )
    parser.add_argument(
        'fit_comps',
        nargs='+', type=int,
        help='The principal components to use in the shape. Numbering starts at 0.'
        )
    parser.add_argument(
        'fit_mode',
        choices=['st', 'ts', 'corr'],
        help='''Registration objective function type. The choices are
st: minimise distance between each source (mean mesh) point and its closest target point 
ts: minimise distance between each target point and its closest source (mean mesh) point
corr: minimise distance between each ordered pair of source (mean mesh) and target points.  
        '''
        )
    parser.add_argument(
        '-t', '--target',
        help='file path of the target mesh or point cloud.'
        )
    parser.add_argument(
        '-o', '--out',
        help='file path of the output registered model.'
        )
    parser.add_argument(
        '-a', '--auto-align',
        action='store_true',
        help='perform initial ICP rigid alignment'
        )
    parser.add_argument(
        '-m', '--mweight',
        type=float, default=1.0,
        help='mahalanobis weight'
        )
    parser.add_argument(
        '-s', '--sample',
        type=int, default=None,
        help='number of points to sample from target for fitting.'
        )
    parser.add_argument(
        '--fit_scale',
        action='store_true',
        help='Fit isotropic scaling independent of shape.'
        )
    parser.add_argument(
        '-p', '--points-only',
        help='''Model are point clouds only. Expected file format is 1 header 
line, then n,x,y,z on each line after. UNTESTED'''
        )
    parser.add_argument(
        '-b', '--batch',
        help='file path of a list of model paths to fit. 1st model on list will be the source.'
        )
    parser.add_argument(
        '-d', '--outdir',
        help='directory path of the output registered models when using batch mode.'
        )
    parser.add_argument(
        '--outext',
        choices=('.obj', '.wrl', '.stl', '.ply', '.vtp'),
        help='output file extension. Ignored if --out is given, useful in batch mode.'
        )
    parser.add_argument(
        '-v', '--view',
        action='store_true',
        help='Visualise measurements and model in 3D'
        )
    parser.add_argument(
        '-l', '--log',
        help='log file'
        )
    args = parser.parse_args()

    # start logging
    if args.log:
        log_fmt = '%(levelname)s - %(asctime)s: %(message)s'
        log_level = logging.INFO

        logging.basicConfig(
            filename=args.log,
            level=log_level,
            format=log_fmt,
            )
        logging.info(
            'Starting shape model registration',
            )

    if args.batch is None:
        reg_single(args)
    else:
        reg_batch(args)
        

if __name__=='__main__':
    main()